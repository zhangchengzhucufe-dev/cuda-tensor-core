// A full pre-norm transformer block forward pass, as one pipeline:
//
//   ln1 = LayerNorm(x)                       S x d
//   qkv = ln1 @ Wqkv + bqkv                  S x 3d
//   attn = softmax(Q K^T / sqrt(d)) V        S x d   (online softmax, fused)
//   x    = x + attn @ Wproj                  residual + GEMM
//   ln2 = LayerNorm(x)
//   h    = gelu(ln2 @ W1 + b1)               S x ffn (bias+gelu fused)
//   x    = x + h @ W2 + b2                   residual
//
// This is the file that ties the other examples together: the tiled GEMM
// from 03, the block reductions from 09, the fused online-softmax attention
// from 10, and the bias+gelu fusion all show up as stages here. Attention
// runs 4 heads of 32 dims each, packed into the qkv buffer -- the per-head
// math is identical, and the concatenated head outputs feed Wproj without
// any reshuffling.
//
// Every stage is timed separately, and the whole thing is verified against
// a plain double-precision CPU implementation. And since the stage timings
// below turn out to be mostly launch overhead at this size, the same
// forward is also captured into a CUDA graph (see 13_graphs) and replayed,
// eager vs graph, at the end.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "error_check.h"

// ---- sizes (all multiples of the GEMM tile, keeps the kernels branch-free) ----
#define S 128     // sequence length
#define D 128     // model dim
#define NH 4      // attention heads (D must be divisible by NH)
#define FFN 512   // mlp hidden dim
#define TILE 32

// ---------- kernels (small variations on the ones from earlier dirs) ----------

// block-wide sum, same helper as 09_nn_ops. Every thread calls, every
// thread gets the result (broadcast via shared memory)
__device__ float block_reduce_sum_tb(float v, float* warp_buf) {
    __syncthreads();
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) warp_buf[warp] = v;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    v = (threadIdx.x < num_warps) ? warp_buf[threadIdx.x] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if (threadIdx.x == 0) warp_buf[0] = v;
    __syncthreads();
    return warp_buf[0];
}

__global__ void layernorm_tb(const float* x, const float* gamma, const float* beta,
                             float* y, int rows, int cols, float eps) {
    __shared__ float warp_buf[32];
    int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    float sum = 0.f, sum_sq = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = xr[j];
        sum += v;
        sum_sq += v * v;
    }
    sum = block_reduce_sum_tb(sum, warp_buf);
    sum_sq = block_reduce_sum_tb(sum_sq, warp_buf);

    if (threadIdx.x == 0) {
        float mean = sum / cols;
        warp_buf[1] = sum_sq / cols - mean * mean;  // stash var in shared too
        warp_buf[2] = mean;
    }
    __syncthreads();
    float rstd = rsqrtf(warp_buf[1] + eps);
    float mean = warp_buf[2];

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        yr[j] = (xr[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

// plain tiled GEMM, straight from 03_gemm
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    float acc = 0.f;
    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[static_cast<size_t>(row) * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[static_cast<size_t>(b_row) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[static_cast<size_t>(row) * N + col] = acc;
}

// Fused online-softmax attention, multi-head. One block per (query, head),
// blockDim.x == head dim. Q/K/V point at their slices of the packed qkv
// buffer (row stride ld); inside a slice, head h owns columns
// [h*dh, (h+1)*dh). Each head writes its Dh outputs contiguously into the
// [S, D] attention output, which is exactly the layout Wproj consumes --
// so going from 1 head to 4 costs nothing in the surrounding GEMMs.
// (param names dodge the size macros above)
__global__ void attention_multihead(const float* Q, const float* K, const float* V,
                                    float* O, int n_seq, int dh, int nh, int ld,
                                    float scale) {
    __shared__ float warp_buf[32];
    int q = blockIdx.x / nh;
    int h = blockIdx.x % nh;
    int dim = threadIdx.x;  // one thread per head dim, blockDim.x == dh

    float qv = Q[static_cast<size_t>(q) * ld + h * dh + dim];
    const float* kcol = K + h * dh + dim;  // column 'dim' of every key row
    const float* vcol = V + h * dh + dim;

    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int k = 0; k < n_seq; ++k) {
        float s = block_reduce_sum_tb(qv * kcol[static_cast<size_t>(k) * ld], warp_buf) * scale;
        float m_new = fmaxf(m, s);
        float p = __expf(m - m_new);
        float e = __expf(s - m_new);
        acc = acc * p + e * vcol[static_cast<size_t>(k) * ld];
        l = l * p + e;
        m = m_new;
    }
    O[static_cast<size_t>(q) * (nh * dh) + h * dh + dim] = acc / l;
}

// out = a + b, the residuals
__global__ void add_rows(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// y = gelu(x + col_bias), the mlp up-projection activation, fused
__global__ void bias_gelu_tb(const float* x, const float* bias, float* y,
                             int n, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + bias[i % cols];
    const float k = 0.7978845608f;
    y[i] = 0.5f * v * (1.f + tanhf(k * (v + 0.044715f * v * v * v)));
}

// y = x + col_bias, plain bias add for qkv / w2 projections
__global__ void bias_add_tb(const float* x, const float* bias, float* y,
                            int n, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + bias[i % cols];
}

// ---------- CPU reference, double precision, dead simple on purpose ----------

static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<double>& C, int M, int N, int K) {
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                       B[static_cast<size_t>(k) * N + j];
            C[static_cast<size_t>(i) * N + j] = acc;
        }
}

static void cpu_block_forward(const std::vector<float>& x,
                              const std::vector<float>& wqkv, const std::vector<float>& bqkv,
                              const std::vector<float>& wproj,
                              const std::vector<float>& g1, const std::vector<float>& bt1,
                              const std::vector<float>& g2, const std::vector<float>& bt2,
                              const std::vector<float>& w1, const std::vector<float>& b1,
                              const std::vector<float>& w2, const std::vector<float>& b2,
                              std::vector<double>& out) {
    std::vector<double> ln1(static_cast<size_t>(S) * D), ln2(static_cast<size_t>(S) * D);
    auto layernorm = [&](const std::vector<float>& in, const std::vector<float>& g,
                         const std::vector<float>& b, std::vector<double>& o) {
        for (int r = 0; r < S; ++r) {
            double mean = 0, var = 0;
            for (int j = 0; j < D; ++j) mean += in[r * D + j];
            mean /= D;
            for (int j = 0; j < D; ++j) {
                double d = in[r * D + j] - mean;
                var += d * d;
            }
            var /= D;
            double rstd = 1.0 / std::sqrt(var + 1e-5);
            for (int j = 0; j < D; ++j)
                o[static_cast<size_t>(r) * D + j] =
                    (in[static_cast<size_t>(r) * D + j] - mean) * rstd * g[j] + b[j];
        }
    };
    layernorm(x, g1, bt1, ln1);

    // qkv = ln1 @ wqkv + bqkv
    std::vector<double> qkv;
    cpu_gemm(std::vector<float>(ln1.begin(), ln1.end()), wqkv, qkv, S, 3 * D, D);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] += bqkv[i % (3 * D)];

    // attention over NH heads of D/NH dims each. The packed qkv layout puts
    // all heads of Q first, then K, then V; head h owns columns
    // [h*dh, (h+1)*dh) inside each slice
    const int dh = D / NH;
    float hscale = 1.f / std::sqrt(static_cast<float>(dh));
    std::vector<double> attn(static_cast<size_t>(S) * D, 0.0);
    std::vector<double> scores(S);
    for (int h = 0; h < NH; ++h) {
        for (int q = 0; q < S; ++q) {
            double m = -INFINITY;
            for (int k = 0; k < S; ++k) {
                double dot = 0.0;
                for (int d = 0; d < dh; ++d)
                    dot += qkv[static_cast<size_t>(q) * (3 * D) + h * dh + d] *        // Q slice
                           qkv[static_cast<size_t>(k) * (3 * D) + D + h * dh + d];    // K slice
                scores[k] = dot * hscale;
                m = std::max(m, scores[k]);
            }
            double denom = 0.0;
            for (int k = 0; k < S; ++k) {
                scores[k] = std::exp(scores[k] - m);
                denom += scores[k];
            }
            for (int d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (int k = 0; k < S; ++k)
                    acc += scores[k] * qkv[static_cast<size_t>(k) * (3 * D) + 2 * D + h * dh + d];  // V slice
                attn[static_cast<size_t>(q) * D + h * dh + d] = acc / denom;
            }
        }
    }

    // x = x + attn @ wproj
    std::vector<double> proj;
    cpu_gemm(std::vector<float>(attn.begin(), attn.end()), wproj, proj, S, D, D);
    std::vector<double> res1(static_cast<size_t>(S) * D);
    for (size_t i = 0; i < res1.size(); ++i) res1[i] = x[i] + proj[i];

    layernorm(std::vector<float>(res1.begin(), res1.end()), g2, bt2, ln2);

    // mlp
    std::vector<double> h;
    cpu_gemm(std::vector<float>(ln2.begin(), ln2.end()), w1, h, S, FFN, D);
    for (size_t i = 0; i < h.size(); ++i) {
        double v = h[i] + b1[i % FFN];
        h[i] = 0.5 * v * (1.0 + std::tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
    }
    std::vector<double> mlp;
    cpu_gemm(std::vector<float>(h.begin(), h.end()), w2, mlp, S, D, FFN);

    out.resize(static_cast<size_t>(S) * D);
    for (size_t i = 0; i < out.size(); ++i) out[i] = res1[i] + mlp[i] + b2[i % D];
}

// ---------- driver ----------

int main() {
    std::srand(0);
    const int dh = D / NH;
    float scale = 1.f / std::sqrt(static_cast<float>(dh));  // per-head scale
    std::printf("transformer block fwd: seq=%d, d=%d, %d heads x %d, ffn=%d (pre-norm)\n",
                S, D, NH, dh, FFN);

    // small weights so activations stay O(1)
    auto rnd = [](size_t n, float lo, float hi) {
        std::vector<float> v(n);
        for (auto& e : v) e = lo + (hi - lo) * (std::rand() / static_cast<float>(RAND_MAX));
        return v;
    };
    std::vector<float> h_x = rnd(static_cast<size_t>(S) * D, -0.5f, 0.5f);
    std::vector<float> h_wqkv = rnd(static_cast<size_t>(D) * 3 * D, -0.05f, 0.05f);
    std::vector<float> h_bqkv = rnd(3 * D, -0.05f, 0.05f);
    std::vector<float> h_wproj = rnd(static_cast<size_t>(D) * D, -0.05f, 0.05f);
    std::vector<float> h_g1 = rnd(D, 0.9f, 1.1f), h_bt1 = rnd(D, -0.1f, 0.1f);
    std::vector<float> h_g2 = rnd(D, 0.9f, 1.1f), h_bt2 = rnd(D, -0.1f, 0.1f);
    std::vector<float> h_w1 = rnd(static_cast<size_t>(D) * FFN, -0.05f, 0.05f);
    std::vector<float> h_b1 = rnd(FFN, -0.05f, 0.05f);
    std::vector<float> h_w2 = rnd(static_cast<size_t>(FFN) * D, -0.05f, 0.05f);
    std::vector<float> h_b2 = rnd(D, -0.05f, 0.05f);

    // CPU reference
    std::vector<double> h_ref;
    cpu_block_forward(h_x, h_wqkv, h_bqkv, h_wproj, h_g1, h_bt1, h_g2, h_bt2,
                      h_w1, h_b1, h_w2, h_b2, h_ref);

    // ---- device buffers ----
    auto dalloc = [](size_t elems) {
        float* p;
        CUDA_CHECK(cudaMalloc(&p, elems * sizeof(float)));
        return p;
    };
    size_t sd = static_cast<size_t>(S) * D, s3d = static_cast<size_t>(S) * 3 * D;
    size_t sf = static_cast<size_t>(S) * FFN;
    float *d_x = dalloc(sd), *d_ln1 = dalloc(sd), *d_qkv = dalloc(s3d);
    float *d_attn = dalloc(sd), *d_tmp = dalloc(sd), *d_res1 = dalloc(sd);
    float *d_ln2 = dalloc(sd), *d_ffn = dalloc(sf), *d_res2 = dalloc(sd);
    float *d_wqkv = dalloc(h_wqkv.size()), *d_bqkv = dalloc(h_bqkv.size());
    float *d_wproj = dalloc(h_wproj.size());
    float *d_g1 = dalloc(D), *d_bt1 = dalloc(D), *d_g2 = dalloc(D), *d_bt2 = dalloc(D);
    float *d_w1 = dalloc(h_w1.size()), *d_b1 = dalloc(h_b1.size());
    float *d_w2 = dalloc(h_w2.size()), *d_b2 = dalloc(h_b2.size());
    // (weights could be __constant__ for the small ones; left as-is for brevity)
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sd * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wqkv, h_wqkv.data(), h_wqkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bqkv, h_bqkv.data(), h_bqkv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wproj, h_wproj.data(), h_wproj.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g1, h_g1.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt1, h_bt1.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g2, h_g2.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt2, h_bt2.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int vec_grid = (sd + block - 1) / block;
    int qkv_grid = (s3d + block - 1) / block;  // qkv is S*3*D, easy to underlaunch here
    int ffn_grid = (sf + block - 1) / block;
    dim3 gemm_block(TILE, TILE);

    CudaTimer timer;
    float t_ln1, t_qkv, t_att, t_res1, t_ln2, t_fc1, t_res2;

    // the actual block forward, stage by stage
    timer.start();
    layernorm_tb<<<S, D>>>(d_x, d_g1, d_bt1, d_ln1, S, D, 1e-5f);
    CHECK_KERNEL_LAUNCH();
    t_ln1 = timer.stop();

    timer.start();
    gemm_tiled<<<dim3(3 * D / TILE, S / TILE), gemm_block>>>(d_ln1, d_wqkv, d_qkv, S, 3 * D, D);
    bias_add_tb<<<qkv_grid, block>>>(d_qkv, d_bqkv, d_qkv, s3d, 3 * D);
    CHECK_KERNEL_LAUNCH();
    t_qkv = timer.stop();

    timer.start();
    // Q/K/V are column slices of d_qkv, all with row stride 3*D
    attention_multihead<<<S * NH, dh>>>(d_qkv, d_qkv + D, d_qkv + 2 * D, d_attn, S, dh, NH, 3 * D, scale);
    CHECK_KERNEL_LAUNCH();
    t_att = timer.stop();

    timer.start();
    gemm_tiled<<<dim3(D / TILE, S / TILE), gemm_block>>>(d_attn, d_wproj, d_tmp, S, D, D);
    add_rows<<<vec_grid, block>>>(d_x, d_tmp, d_res1, sd);
    CHECK_KERNEL_LAUNCH();
    t_res1 = timer.stop();

    timer.start();
    layernorm_tb<<<S, D>>>(d_res1, d_g2, d_bt2, d_ln2, S, D, 1e-5f);
    CHECK_KERNEL_LAUNCH();
    t_ln2 = timer.stop();

    timer.start();
    gemm_tiled<<<dim3(FFN / TILE, S / TILE), gemm_block>>>(d_ln2, d_w1, d_ffn, S, FFN, D);
    bias_gelu_tb<<<ffn_grid, block>>>(d_ffn, d_b1, d_ffn, sf, FFN);
    CHECK_KERNEL_LAUNCH();
    t_fc1 = timer.stop();

    timer.start();
    gemm_tiled<<<dim3(D / TILE, S / TILE), gemm_block>>>(d_ffn, d_w2, d_tmp, S, D, FFN);
    bias_add_tb<<<vec_grid, block>>>(d_tmp, d_b2, d_tmp, sd, D);
    add_rows<<<vec_grid, block>>>(d_res1, d_tmp, d_res2, sd);
    CHECK_KERNEL_LAUNCH();
    t_res2 = timer.stop();

    float total = t_ln1 + t_qkv + t_att + t_res1 + t_ln2 + t_fc1 + t_res2;
    std::printf("stage timings:\n");
    std::printf("  ln1 %.3f | qkv gemm %.3f | attention %.3f | proj+res %.3f\n",
                t_ln1, t_qkv, t_att, t_res1);
    std::printf("  ln2 %.3f | fc1+gelu %.3f | fc2+res %.3f | total %.3f ms\n",
                t_ln2, t_fc1, t_res2, total);
    std::printf("(ln1 includes first-launch overhead; compare with ln2 for the\n"
                " real cost. at this size the stages are microseconds and launch\n"
                " gaps matter more than the kernels -- exactly the cuda-graphs case)\n");

    // verify
    std::vector<float> h_out(sd);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_res2, sd * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_out.size(); ++i) {
        double limit = 2e-3 * (1.0 + std::fabs(h_ref[i]));
        if (std::fabs(h_out[i] - h_ref[i]) > limit) {
            std::fprintf(stderr, "FAILED at %zu: got %f, expect %f\n", i, h_out[i],
                         h_ref[i]);
            return 1;
        }
    }
    std::printf("verification passed against the double-precision CPU block\n");

    // ---- the same forward as one cuda graph ----
    // the stage timings above are mostly launch gaps at this size, so do
    // what inference engines do for small decode steps: capture the whole
    // sequence once, then submit it with one call per iteration
    const int ITERS = 50;
    auto run_block = [&](cudaStream_t stream) {
        layernorm_tb<<<S, D, 0, stream>>>(d_x, d_g1, d_bt1, d_ln1, S, D, 1e-5f);
        gemm_tiled<<<dim3(3 * D / TILE, S / TILE), gemm_block, 0, stream>>>(
            d_ln1, d_wqkv, d_qkv, S, 3 * D, D);
        bias_add_tb<<<qkv_grid, block, 0, stream>>>(d_qkv, d_bqkv, d_qkv, s3d, 3 * D);
        attention_multihead<<<S * NH, dh, 0, stream>>>(d_qkv, d_qkv + D, d_qkv + 2 * D,
                                                       d_attn, S, dh, NH, 3 * D, scale);
        gemm_tiled<<<dim3(D / TILE, S / TILE), gemm_block, 0, stream>>>(
            d_attn, d_wproj, d_tmp, S, D, D);
        add_rows<<<vec_grid, block, 0, stream>>>(d_x, d_tmp, d_res1, sd);
        layernorm_tb<<<S, D, 0, stream>>>(d_res1, d_g2, d_bt2, d_ln2, S, D, 1e-5f);
        gemm_tiled<<<dim3(FFN / TILE, S / TILE), gemm_block, 0, stream>>>(
            d_ln2, d_w1, d_ffn, S, FFN, D);
        bias_gelu_tb<<<ffn_grid, block, 0, stream>>>(d_ffn, d_b1, d_ffn, sf, FFN);
        gemm_tiled<<<dim3(D / TILE, S / TILE), gemm_block, 0, stream>>>(
            d_ffn, d_w2, d_tmp, S, D, FFN);
        bias_add_tb<<<vec_grid, block, 0, stream>>>(d_tmp, d_b2, d_tmp, sd, D);
        add_rows<<<vec_grid, block, 0, stream>>>(d_res1, d_tmp, d_res2, sd);
    };

    // eager: the whole 12-launch sequence, ITERS times
    timer.start();
    for (int i = 0; i < ITERS; ++i) run_block(nullptr);
    CHECK_KERNEL_LAUNCH();
    float ms_eager = timer.stop();

    // capture once, replay ITERS times
    cudaStream_t cs;
    CUDA_CHECK(cudaStreamCreate(&cs));
    CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
    run_block(cs);
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(cs, &graph));
    cudaGraphExec_t gexec;
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));

    timer.start();
    for (int i = 0; i < ITERS; ++i) CUDA_CHECK(cudaGraphLaunch(gexec, nullptr));
    float ms_graph = timer.stop();

    std::printf("whole block x%d: eager %.3f ms, graph %.3f ms (%.2fx)\n",
                ITERS, ms_eager, ms_graph, ms_eager / ms_graph);
    std::printf("only a modest win here: these kernels do enough real work\n"
                " (~300 us/iter on the GPU) that the CPU mostly keeps up even\n"
                " eager. the 13_graphs demo with its tiny kernels is\n"
                " where capture pays properly -- same mechanism, different ratio\n");

    // graph replays rewrote d_res2; inputs are unchanged so it must match again
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_res2, sd * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_out.size(); ++i) {
        double limit = 2e-3 * (1.0 + std::fabs(h_ref[i]));
        if (std::fabs(h_out[i] - h_ref[i]) > limit) {
            std::fprintf(stderr, "graph replay FAILED at %zu\n", i);
            return 1;
        }
    }
    std::printf("graph replay output still matches the reference\n");

    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cs);
    return 0;
}
