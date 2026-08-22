// LayerNorm. Standard transformer ingredient, extremely common interview
// question: y = (x - mean) / sqrt(var + eps) * gamma + beta, per row,
// with the biased variance.
//
// Implementation notes:
//   - one block per row, two block reductions: mean first, then the second
//     moment. E[x^2] - mean^2 gets the variance in the same sweep (one less
//     pass over the row). Numerically it's not as solid as Welford, but for
//     normal-scale inputs it's fine -- a tradeoff, not an oversight
//   - gamma/beta are per-column affine params, read straight from global
//     memory (L2 hit rate is great since every row reads the same ones)
//   - rsqrtf instead of 1/sqrtf: it's the hardware approximation, fast and
//     plenty accurate here
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define BLOCK 256

// same helper as softmax_rowwise: warp shuffle -> shared memory -> broadcast
// through shared memory (a __shfl_sync broadcast can't cross warps)
__device__ float block_reduce_sum(float v, float* warp_buf) {
    __syncthreads();  // let the previous call's broadcast reads finish
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

__global__ void layernorm(const float* x, const float* gamma, const float* beta,
                          float* y, int rows, int cols, float eps) {
    __shared__ float warp_buf[BLOCK / 32];
    __shared__ float s_mean, s_rstd;

    int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    // one reduction pass collects both sum(x) and sum(x^2)
    float sum = 0.f, sum_sq = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = xr[j];
        sum += v;
        sum_sq += v * v;
    }
    sum = block_reduce_sum(sum, warp_buf);
    sum_sq = block_reduce_sum(sum_sq, warp_buf);

    if (threadIdx.x == 0) {
        float mean = sum / cols;
        float var = sum_sq / cols - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean;
    float rstd = s_rstd;

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        yr[j] = (xr[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int rows = 4096, cols = 1024;
    if (argc > 2) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    const float eps = 1e-5f;
    std::printf("layernorm: %d rows x %d cols\n", rows, cols);

    size_t elems = static_cast<size_t>(rows) * cols;
    size_t bytes = elems * sizeof(float);
    float* h_x = static_cast<float*>(std::malloc(bytes));
    float* h_y = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    float* h_gamma = static_cast<float*>(std::malloc(cols * sizeof(float)));
    float* h_beta = static_cast<float*>(std::malloc(cols * sizeof(float)));
    fill_random_host(h_x, elems, -2.f, 2.f);
    fill_random_host(h_gamma, cols, 0.5f, 1.5f);
    fill_random_host(h_beta, cols, -0.5f, 0.5f);

    // CPU reference in double
    for (int r = 0; r < rows; ++r) {
        const float* xr = h_x + static_cast<size_t>(r) * cols;
        float* yr = h_ref + static_cast<size_t>(r) * cols;
        double mean = 0.0, mean_sq = 0.0;
        for (int j = 0; j < cols; ++j) {
            mean += xr[j];
            mean_sq += static_cast<double>(xr[j]) * xr[j];
        }
        mean /= cols;
        mean_sq /= cols;
        double var = mean_sq - mean * mean;
        double rstd = 1.0 / std::sqrt(var + eps);
        for (int j = 0; j < cols; ++j)
            yr[j] = static_cast<float>((xr[j] - mean) * rstd * h_gamma[j] + h_beta[j]);
    }

    float *d_x, *d_y, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, cols * sizeof(float),
                          cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.start();
    layernorm<<<rows, BLOCK>>>(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    std::printf("kernel: %.3f ms\n", ms);
    if (!compare_close(h_y, h_ref, elems, 1e-4f)) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("verification passed\n");

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);
    std::free(h_gamma);
    std::free(h_beta);
    return 0;
}
