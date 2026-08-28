// Single-head attention forward, fused, with online softmax.
// This is the numerical core of Flash Attention, in its simplest form.
//
//   O = softmax(Q K^T / sqrt(d)) V,   Q,K,V: S x D
//
// The classic three-pass implementation materializes the S x S score
// matrix in memory (write it, read it for softmax, write that, read it
// again for the V multiply) -- and it grows quadratically with sequence
// length. Flash Attention's trick is to run the softmax online:
//   while streaming over K/V, keep the running max m and exp-sum l, and
//   for each new score s:
//     m_new = max(m, s)
//     acc   = acc * exp(m - m_new) + exp(s - m_new) * v_j   (rescale the old result)
//     l     = l     * exp(m - m_new) + exp(s - m_new)
// Mathematically identical to the full softmax, but the scores never
// touch memory.
//
// Teaching version here: one block per query row, D=64 threads, each thread
// owning one output dimension's accumulator. Real FlashAttention also tiles
// over keys, stages K/V tiles in shared memory, and splits work across
// warps -- same structure, finer grain.
//
// Memory story (S rows, D dims):
//   three-pass: an extra S x S score matrix in memory
//   this one:   zero extra memory, O(D) registers + a couple of shared floats
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "error_check.h"

#define D 64  // head dim; this kernel requires D == blockDim.x (64 = 2 warps)

// dot product over the block (2 warps here), result broadcast to everyone
__device__ __forceinline__ float block_dot_64(float v, float* warp_buf) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) warp_buf[warp] = v;
    __syncthreads();
    float total = warp_buf[0] + warp_buf[1];
    __syncthreads();  // nobody overwrites warp_buf before it's been read
    return total;
}

__global__ void attention_forward(const float* Q, const float* K, const float* V,
                                  float* O, int S, float scale) {
    __shared__ float warp_buf[D / 32];
    int q = blockIdx.x;      // query row this block handles
    int dim = threadIdx.x;   // output dimension owned by this thread

    float qv = Q[static_cast<size_t>(q) * D + dim];

    // online softmax state: running max, exp-sum, and this dim's accumulator
    float m = -INFINITY;
    float l = 0.f;
    float acc = 0.f;

    for (int k = 0; k < S; ++k) {
        // score(q,k) = dot(Q[q], K[k]) * scale, all threads cooperate
        const float* kr = K + static_cast<size_t>(k) * D;
        float s = block_dot_64(qv * kr[dim], warp_buf) * scale;

        // online softmax update: the old accumulator gets rescaled by
        // exp(m - m_new) before the new term comes in
        float m_new = fmaxf(m, s);
        float p = __expf(m - m_new);   // rescale factor (0 on the very first key)
        float e = __expf(s - m_new);   // weight of the new term
        acc = acc * p + e * V[static_cast<size_t>(k) * D + dim];
        l = l * p + e;
        m = m_new;
    }
    O[static_cast<size_t>(q) * D + dim] = acc / l;
}

// CPU reference, three-pass style (full score matrix, then softmax, then V)
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int S, float scale) {
    std::vector<float> scores(S);
    for (int q = 0; q < S; ++q) {
        float m = -INFINITY;
        for (int k = 0; k < S; ++k) {
            float dot = 0.f;
            for (int d = 0; d < D; ++d)
                dot += Q[static_cast<size_t>(q) * D + d] *
                       K[static_cast<size_t>(k) * D + d];
            scores[k] = dot * scale;
            m = fmaxf(m, scores[k]);
        }
        float denom = 0.f;
        for (int k = 0; k < S; ++k) {
            scores[k] = std::exp(scores[k] - m);
            denom += scores[k];
        }
        for (int d = 0; d < D; ++d) {
            float acc = 0.f;
            for (int k = 0; k < S; ++k)
                acc += scores[k] * V[static_cast<size_t>(k) * D + d];
            O[static_cast<size_t>(q) * D + d] = acc / denom;
        }
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int S = 512;  // sequence length
    if (argc > 1) S = std::atoi(argv[1]);
    float scale = 1.f / std::sqrt(static_cast<float>(D));
    std::printf("single-head attention: seq=%d, head_dim=%d (fused, online softmax)\n", S, D);

    size_t qkv_bytes = static_cast<size_t>(S) * D * sizeof(float);
    float* h_q = static_cast<float*>(std::malloc(qkv_bytes));
    float* h_k = static_cast<float*>(std::malloc(qkv_bytes));
    float* h_v = static_cast<float*>(std::malloc(qkv_bytes));
    float* h_o = static_cast<float*>(std::malloc(qkv_bytes));
    float* h_ref = static_cast<float*>(std::malloc(qkv_bytes));
    fill_random_host(h_q, static_cast<size_t>(S) * D, -2.f, 2.f);
    fill_random_host(h_k, static_cast<size_t>(S) * D, -2.f, 2.f);
    fill_random_host(h_v, static_cast<size_t>(S) * D, -2.f, 2.f);

    attention_cpu(h_q, h_k, h_v, h_ref, S, scale);

    float *d_q, *d_k, *d_v, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_k, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_v, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_o, qkv_bytes));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, qkv_bytes, cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.start();
    attention_forward<<<S, D>>>(d_q, d_k, d_v, d_o, S, scale);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_o, d_o, qkv_bytes, cudaMemcpyDeviceToHost));

    std::printf("kernel: %.3f ms (no S x S score matrix ever exists in memory)\n", ms);
    if (!compare_close(h_o, h_ref, static_cast<size_t>(S) * D, 1e-4f)) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("verification passed (online softmax is mathematically the same as three-pass)\n");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    std::free(h_q);
    std::free(h_k);
    std::free(h_v);
    std::free(h_o);
    std::free(h_ref);
    return 0;
}
