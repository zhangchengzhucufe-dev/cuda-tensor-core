// Row-wise softmax. Shows up in every transformer (attention rows, the
// final output layer), and it's a frequent "write it from memory" interview
// question.
//   y[i][j] = exp(x[i][j] - max_j) / sum_j exp(x[i][j] - max_j)
// Subtracting the row max is not optional -- without it a large input
// overflows exp. That's the "safe softmax" trick.
//
// Two kernels, same math:
//   v1: one thread per row. Shortest code, but zero parallelism within a
//       row and it sweeps the row three times (max, sum, write). Gets ugly
//       as rows get wide.
//   v2: one block per row. Three strided passes, each ending in a block
//       reduction. This is how real inference engines write it.
//
// block_reduce_sum/max below are the reusable piece -- warp shuffle plus
// one shared memory hop. Layernorm uses the same pattern.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define BLOCK 256

// Block-wide sum. Warp shuffles first, warp partials through shared memory.
// EVERY thread must call this (it has __syncthreads inside) and every
// thread gets a valid return value -- the broadcast at the end goes through
// shared memory because __shfl_sync only crosses lanes, not warps. That
// distinction cost me a debug session, hence the comment.
__device__ float block_reduce_sum(float v, float* warp_buf) {
    __syncthreads();  // make sure the previous call's broadcast reads are done
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

// Block-wide max, same structure
__device__ float block_reduce_max(float v, float* warp_buf) {
    __syncthreads();
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    if (lane == 0) warp_buf[warp] = v;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    v = (threadIdx.x < num_warps) ? warp_buf[threadIdx.x] : -INFINITY;
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    if (threadIdx.x == 0) warp_buf[0] = v;
    __syncthreads();
    return warp_buf[0];
}

// v1: one thread per row, three serial sweeps
__global__ void softmax_thread_per_row(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    float m = -INFINITY;
    for (int j = 0; j < cols; ++j) m = fmaxf(m, xr[j]);
    float s = 0.f;
    for (int j = 0; j < cols; ++j) s += __expf(xr[j] - m);
    for (int j = 0; j < cols; ++j) yr[j] = __expf(xr[j] - m) / s;
}

// v2: one block per row
__global__ void softmax_block_per_row(const float* x, float* y, int rows, int cols) {
    __shared__ float warp_buf[BLOCK / 32];
    int row = blockIdx.x;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float* yr = y + static_cast<size_t>(row) * cols;

    // pass 1: row max (return value is valid in every thread)
    float v = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        v = fmaxf(v, xr[j]);
    float m = block_reduce_max(v, warp_buf);

    // pass 2: sum(exp)
    float sum = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        sum += __expf(xr[j] - m);
    sum = block_reduce_sum(sum, warp_buf);

    // pass 3: normalize and write
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        yr[j] = __expf(xr[j] - m) / sum;
}

int main(int argc, char** argv) {
    std::srand(0);
    int rows = 4096, cols = 1024;
    if (argc > 2) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    std::printf("softmax: %d rows x %d cols\n", rows, cols);

    size_t elems = static_cast<size_t>(rows) * cols;
    size_t bytes = elems * sizeof(float);
    float* h_x = static_cast<float*>(std::malloc(bytes));
    float* h_y = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    // inputs pushed up to +-30 to prove the max-subtraction
    // actually keeps things stable
    fill_random_host(h_x, elems, -30.f, 30.f);

    for (int r = 0; r < rows; ++r) {
        const float* xr = h_x + static_cast<size_t>(r) * cols;
        float* yr = h_ref + static_cast<size_t>(r) * cols;
        float m = -INFINITY;
        for (int j = 0; j < cols; ++j) m = fmaxf(m, xr[j]);
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += std::exp(xr[j] - m);
        for (int j = 0; j < cols; ++j) yr[j] = static_cast<float>(std::exp(xr[j] - m) / s);
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    CudaTimer timer;

    // v1
    timer.start();
    softmax_thread_per_row<<<(rows + 255) / 256, 256>>>(d_x, d_y, rows, cols);
    CHECK_KERNEL_LAUNCH();
    float ms1 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, elems, 1e-4f)) {
        std::printf("thread-per-row FAILED\n");
        return 1;
    }

    // v2
    timer.start();
    softmax_block_per_row<<<rows, BLOCK>>>(d_x, d_y, rows, cols);
    CHECK_KERNEL_LAUNCH();
    float ms2 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, elems, 1e-4f)) {
        std::printf("block-per-row FAILED\n");
        return 1;
    }

    std::printf("thread-per-row: %8.3f ms\n", ms1);
    std::printf("block-per-row : %8.3f ms  (block reduction version)\n", ms2);
    std::printf("both passed (no overflow even at +-30 inputs, thanks to the max subtraction)\n");

    cudaFree(d_x);
    cudaFree(d_y);
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);
    return 0;
}
