// Decode-phase attention with a KV cache -- the shape real LLM inference
// actually runs, as opposed to the prefill shape of attention_forward.cu.
//
// In prefill you process S query tokens at once and the S x S score matrix
// is the problem. In decode you generate ONE token per step, and the
// question changes: every past token's K and V are needed again, so you
// cache them instead of recomputing. Per generated token you then do:
//
//   out = softmax(q * K_cache^T / sqrt(d)) * V_cache
//
// with a single query row. Memory per step is O(T x H x D) for the cache
// read -- linear in context, not quadratic, and there is no S x S score
// matrix at all. The cache itself is the point: this file demonstrates the
// append (new K/V lands in the next cache slot) and then the attention
// over [0, T], which is the loop structure of every autoregressive
// sampler.
//
// H=8 heads, D=64. Three kernels, a measured ladder of their own:
//
//   decode_attention   one block per head. The teaching version, and a
//                      lesson in under-parallelism: 8 blocks of 4 warps
//                      leave 22 of 30 SMs empty, and no amount of loop
//                      unrolling hides a 600-cycle DRAM latency with that
//                      little in flight. Measured 24-28 GB/s here.
//
//   decode_partial +   the flash-decoding structure: split the cached
//   decode_combine     tokens across many blocks (each computes scores,
//                      a local max/sum and a local weighted-V over its
//                      chunk), then a tiny second kernel merges the
//                      partials with the exp(m_c - m*) rescale from
//                      attention_forward.cu. Same math, NCHUNKS times
//                      more parallelism: 166 GB/s at a 4K cache and 312
//                      GB/s -- ~93% of what this card can read -- at 16K.
//                      Both paths verify against the same CPU reference.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "error_check.h"

#define HDIM 64
#define NHEADS 8
#define CHUNK 512  // cached tokens per block in the split version

// ---------------------------------------------------------------------------
// teaching version: one block per head, three passes
// ---------------------------------------------------------------------------
__global__ void decode_attention(const float* __restrict__ q,
                                 const float* __restrict__ k_cache,
                                 const float* __restrict__ v_cache, int T,
                                 float* __restrict__ out) {
    int h = blockIdx.x;
    // scores T floats + cross-warp reduce scratch + [max, denom] slots
    extern __shared__ float sm[];
    float* s = sm;        // [T]
    float* red = sm + T;  // [8] cross-warp scratch
    float* brd = red + 8; // [2] max, denom

    const float* qh = q + h * HDIM;
    const float* kh = k_cache + static_cast<size_t>(h) * HDIM;
    const float* vh = v_cache + static_cast<size_t>(h) * HDIM;

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    // pass 1: scores + max. Each warp scores its own tokens (warp w takes
    // t = w, w+4, ...; one lane folds two dims, one shuffle reduce = one
    // full dot), warps write disjoint s[t] slots, so no barrier in here.
    float m = -INFINITY;
    int nw = blockDim.x / 32;
#pragma unroll 4
    for (int t = warp; t < T; t += nw) {
        const float* kt = kh + static_cast<size_t>(t) * NHEADS * HDIM;
        float part = 0.f;
        if (lane * 2 < HDIM) {
            part = qh[lane * 2] * kt[lane * 2];
            part += qh[lane * 2 + 1] * kt[lane * 2 + 1];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_down_sync(0xffffffff, part, off);
        part *= rsqrtf(static_cast<float>(HDIM));
        if (lane == 0) s[t] = part;
        m = fmaxf(m, part);
    }
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float bmax = red[0];
        for (int w = 1; w < nw; ++w) bmax = fmaxf(bmax, red[w]);
        brd[0] = bmax;
    }
    __syncthreads();

    // pass 2a: softmax denominator
    float sum = 0.f;
    for (int t = tid; t < T; t += blockDim.x) sum += __expf(s[t] - brd[0]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.f;
        for (int w = 0; w < nw; ++w) total += red[w];
        brd[1] = total;
    }
    __syncthreads();

    // pass 2b: weighted V sum. 128 threads, two per output dim (each covers
    // half the tokens), four independent accumulator chains so the V loads
    // overlap. First version used 64 threads on one serial chain.
    __shared__ float partial[2][HDIM];
    if (tid < HDIM * 2) {
        int d = tid % HDIM;
        int half = tid / HDIM;
        int lo = half * ((T + 1) / 2);
        int hi = min(T, lo + (T + 1) / 2);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int t = lo;
        for (; t + 3 < hi; t += 4) {
            // loads first: issue all four, let the exps compute while
            // they're in flight
            float v0 = vh[static_cast<size_t>(t) * NHEADS * HDIM + d];
            float v1 = vh[static_cast<size_t>(t + 1) * NHEADS * HDIM + d];
            float v2 = vh[static_cast<size_t>(t + 2) * NHEADS * HDIM + d];
            float v3 = vh[static_cast<size_t>(t + 3) * NHEADS * HDIM + d];
            a0 += __expf(s[t] - brd[0]) * v0;
            a1 += __expf(s[t + 1] - brd[0]) * v1;
            a2 += __expf(s[t + 2] - brd[0]) * v2;
            a3 += __expf(s[t + 3] - brd[0]) * v3;
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; t < hi; ++t)
            acc += __expf(s[t] - brd[0]) *
                   vh[static_cast<size_t>(t) * NHEADS * HDIM + d];
        partial[half][d] = acc;
    }
    __syncthreads();
    if (tid < HDIM)
        out[h * HDIM + tid] = (partial[0][tid] + partial[1][tid]) / brd[1];
}

// ---------------------------------------------------------------------------
// split version: partials over CHUNK-sized token ranges, then a merge
// ---------------------------------------------------------------------------
__global__ void decode_partial(const float* __restrict__ q,
                               const float* __restrict__ k_cache,
                               const float* __restrict__ v_cache, int T,
                               float* __restrict__ p_m, float* __restrict__ p_l,
                               float* __restrict__ p_acc) {
    int h = blockIdx.x;
    int cstart = blockIdx.y * CHUNK;
    int cend = min(T, cstart + CHUNK);
    int len = cend - cstart;

    __shared__ float s[CHUNK];
    __shared__ float red[8];
    __shared__ float brd[2];

    const float* qh = q + h * HDIM;
    const float* kh = k_cache + static_cast<size_t>(h) * HDIM;
    const float* vh = v_cache + static_cast<size_t>(h) * HDIM;

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    int nw = blockDim.x / 32;

    // same three passes as decode_attention, just over [cstart, cend)
    float m = -INFINITY;
#pragma unroll 4
    for (int t = cstart + warp; t < cend; t += nw) {
        const float* kt = kh + static_cast<size_t>(t) * NHEADS * HDIM;
        float part = 0.f;
        if (lane * 2 < HDIM) {
            part = qh[lane * 2] * kt[lane * 2];
            part += qh[lane * 2 + 1] * kt[lane * 2 + 1];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_down_sync(0xffffffff, part, off);
        part *= rsqrtf(static_cast<float>(HDIM));
        if (lane == 0) s[t - cstart] = part;
        m = fmaxf(m, part);
    }
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float bmax = red[0];
        for (int w = 1; w < nw; ++w) bmax = fmaxf(bmax, red[w]);
        brd[0] = bmax;
    }
    __syncthreads();

    float sum = 0.f;
    for (int t = tid; t < len; t += blockDim.x) sum += __expf(s[t] - brd[0]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.f;
        for (int w = 0; w < nw; ++w) total += red[w];
        brd[1] = total;
    }
    __syncthreads();

    __shared__ float partial[2][HDIM];
    if (tid < HDIM * 2) {
        int d = tid % HDIM;
        int half = tid / HDIM;
        int lo = half * ((len + 1) / 2);
        int hi = min(len, lo + (len + 1) / 2);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int t = lo;
        for (; t + 3 < hi; t += 4) {
            float v0 = vh[static_cast<size_t>(cstart + t) * NHEADS * HDIM + d];
            float v1 =
                vh[static_cast<size_t>(cstart + t + 1) * NHEADS * HDIM + d];
            float v2 =
                vh[static_cast<size_t>(cstart + t + 2) * NHEADS * HDIM + d];
            float v3 =
                vh[static_cast<size_t>(cstart + t + 3) * NHEADS * HDIM + d];
            a0 += __expf(s[t] - brd[0]) * v0;
            a1 += __expf(s[t + 1] - brd[0]) * v1;
            a2 += __expf(s[t + 2] - brd[0]) * v2;
            a3 += __expf(s[t + 3] - brd[0]) * v3;
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; t < hi; ++t)
            acc += __expf(s[t] - brd[0]) *
                   vh[static_cast<size_t>(cstart + t) * NHEADS * HDIM + d];
        partial[half][d] = acc;
    }
    __syncthreads();

    size_t cid = static_cast<size_t>(h) * gridDim.y + blockIdx.y;
    if (tid < HDIM) p_acc[cid * HDIM + tid] = partial[0][tid] + partial[1][tid];
    if (tid == 0) {
        p_m[cid] = brd[0];
        p_l[cid] = brd[1];
    }
}

__global__ void decode_combine(const float* __restrict__ p_m,
                               const float* __restrict__ p_l,
                               const float* __restrict__ p_acc, int nc,
                               float* __restrict__ out) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float brd[1];
    if (tid == 0) {
        float mx = -INFINITY;
        for (int c = 0; c < nc; ++c) mx = fmaxf(mx, p_m[h * nc + c]);
        brd[0] = mx;
    }
    __syncthreads();
    // every thread re-walks the tiny list of chunk weights; cheaper and
    // simpler than broadcasting
    if (tid < HDIM) {
        float lsum = 0.f, asum = 0.f;
        for (int c = 0; c < nc; ++c) {
            float w = __expf(p_m[h * nc + c] - brd[0]);
            lsum += w * p_l[h * nc + c];
            asum += w * p_acc[static_cast<size_t>(h * nc + c) * HDIM + tid];
        }
        out[h * HDIM + tid] = asum / lsum;
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::srand(0);
    int T = 4096;  // cached tokens before this step
    if (argc > 1) T = std::atoi(argv[1]);

    size_t cache_elems = static_cast<size_t>(T + 1) * NHEADS * HDIM;
    std::vector<float> h_k(cache_elems), h_v(cache_elems), h_q(NHEADS * HDIM);
    auto rnd = [](std::vector<float>& v) {
        for (auto& x : v)
            x = -1.f + 2.f * (std::rand() / static_cast<float>(RAND_MAX));
    };
    rnd(h_k);
    rnd(h_v);
    rnd(h_q);

    float *d_k, *d_v, *d_q, *d_out, *d_out_split, *d_knew, *d_vnew;
    CUDA_CHECK(cudaMalloc(&d_k, cache_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, cache_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, NHEADS * HDIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_split, NHEADS * HDIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_knew, NHEADS * HDIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vnew, NHEADS * HDIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), cache_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), cache_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- the append: the new token's K/V lands in cache slot T. In a real
    // engine this slot was written when the *previous* token was decoded;
    // the cache is state, not a per-step input. ----
    std::vector<float> h_knew(NHEADS * HDIM), h_vnew(NHEADS * HDIM);
    rnd(h_knew);
    rnd(h_vnew);
    CUDA_CHECK(cudaMemcpy(d_knew, h_knew.data(), h_knew.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vnew, h_vnew.data(), h_vnew.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    size_t slot = static_cast<size_t>(T) * NHEADS * HDIM;
    CUDA_CHECK(cudaMemcpy(d_k + slot, d_knew, NHEADS * HDIM * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_v + slot, d_vnew, NHEADS * HDIM * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    // the CPU model must see the same cache state as the GPU
    std::copy(h_knew.begin(), h_knew.end(), h_k.begin() + slot);
    std::copy(h_vnew.begin(), h_vnew.end(), h_v.begin() + slot);

    int S = T + 1;  // now attending to T past tokens + the new one
    std::printf("decode attention with KV cache: T=%d cached, H=%d heads, D=%d\n",
                S, NHEADS, HDIM);

    int nc = (S + CHUNK - 1) / CHUNK;
    float *p_m, *p_l, *p_acc;
    CUDA_CHECK(cudaMalloc(&p_m, (size_t)NHEADS * nc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_l, (size_t)NHEADS * nc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&p_acc, (size_t)NHEADS * nc * HDIM * sizeof(float)));

    CudaTimer timer;

    // ---- teaching version: one block per head ----
    {
        int max_smem = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(
            &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        size_t smem = (S + 8 + 2) * sizeof(float);
        if (smem > (size_t)max_smem) {
            std::printf("T too big for one-block-per-head smem scores\n");
            return 1;
        }
        // over 48 KB of dynamic smem needs an explicit opt-in per kernel.
        // Asking for the device max outright fails -- static smem and the
        // per-block reservation come off the top -- so ask for exactly
        // what the kernel needs. My first guess was the device max.
        CUDA_CHECK(cudaFuncSetAttribute(
            decode_attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)));
        float ms = 1e30f;
        for (int rep = 0; rep < 50; ++rep) {
            timer.start();
            decode_attention<<<NHEADS, 128, smem>>>(d_q, d_k, d_v, S, d_out);
            CHECK_KERNEL_LAUNCH();
            ms = fminf(ms, timer.stop());
        }
        double mb = 2.0 * S * NHEADS * HDIM * 4 / 1e6;
        std::printf("one block per head : %8.4f ms, %6.1f GB/s (of ~300 the"
                    " card can do)\n", ms, mb / (ms / 1e3) / 1e3);
    }

    // ---- split version: flash-decoding structure ----
    {
        float ms = 1e30f;
        for (int rep = 0; rep < 50; ++rep) {
            timer.start();
            decode_partial<<<dim3(NHEADS, nc), 128>>>(d_q, d_k, d_v, S, p_m,
                                                      p_l, p_acc);
            decode_combine<<<NHEADS, 128>>>(p_m, p_l, p_acc, nc,
                                            d_out_split);
            CHECK_KERNEL_LAUNCH();
            ms = fminf(ms, timer.stop());
        }
        double mb = 2.0 * S * NHEADS * HDIM * 4 / 1e6;
        std::printf("split-T (%3d chunks): %8.4f ms, %6.1f GB/s\n", nc, ms,
                    mb / (ms / 1e3) / 1e3);
    }

    // ---- verify both against the CPU, including the appended slot ----
    std::vector<float> h_out(NHEADS * HDIM), h_out2(NHEADS * HDIM),
        ref(NHEADS * HDIM);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out2.data(), d_out_split,
                          h_out2.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int h = 0; h < NHEADS; ++h) {
        std::vector<float> s(S);
        float m = -1e30f;
        for (int t = 0; t < S; ++t) {
            float dot = 0.f;
            for (int d = 0; d < HDIM; ++d)
                dot += h_q[h * HDIM + d] *
                       h_k[static_cast<size_t>(t) * NHEADS * HDIM + h * HDIM + d];
            s[t] = dot / std::sqrt(static_cast<float>(HDIM));
            m = std::fmax(m, s[t]);
        }
        float denom = 0.f;
        for (int t = 0; t < S; ++t) {
            s[t] = std::exp(s[t] - m);
            denom += s[t];
        }
        for (int d = 0; d < HDIM; ++d) {
            float acc = 0.f;
            for (int t = 0; t < S; ++t)
                acc += s[t] * h_v[static_cast<size_t>(t) * NHEADS * HDIM +
                                  h * HDIM + d];
            ref[h * HDIM + d] = acc / denom;
        }
    }
    if (!compare_close(h_out.data(), ref.data(), ref.size(), 1e-4f) ||
        !compare_close(h_out2.data(), ref.data(), ref.size(), 1e-4f)) {
        std::printf("decode disagrees with CPU!\n");
        return 1;
    }
    std::printf("both match the CPU reference\n");

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_out_split);
    cudaFree(d_knew);
    cudaFree(d_vnew);
    cudaFree(p_m);
    cudaFree(p_l);
    cudaFree(p_acc);
    return 0;
}
