// Kernel fusion 101: bias + GELU as two kernels vs one fused kernel.
// GELU (tanh approximation): gelu(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
//
// Unfused, the intermediate y1 makes a full round trip through memory:
//   kernel1: y1 = x + b        1 read + 1 write
//   kernel2: y2 = gelu(y1)     1 read + 1 write
// Fused: 1 read + 1 write, and no intermediate allocation at all. For
// memory-bound elementwise ops that's close to 2x. This is the most basic
// reason inference engines (TensorRT, vLLM, ...) fuse ops at the graph level.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

__device__ __forceinline__ float gelu(float v) {
    const float k = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * v * (1.f + tanhf(k * (v + 0.044715f * v * v * v)));
}

// v1: two kernels, intermediate lands in global memory
__global__ void add_bias(const float* x, const float* bias, float* y1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y1[i] = x[i] + bias[i % 1024];
}

__global__ void gelu_inplace(const float* y1, float* y2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y2[i] = gelu(y1[i]);
}

// v2: one kernel, intermediate stays in a register
__global__ void bias_gelu_fused(const float* x, const float* bias, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gelu(x[i] + bias[i % 1024]);
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 24;
    if (argc > 1) n = std::atoi(argv[1]);
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::printf("bias+GELU: n = %d (%.0f MB)\n", n, bytes / 1e6);

    float* h_x = static_cast<float*>(std::malloc(bytes));
    float* h_y = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    float h_bias[1024];
    fill_random_host(h_x, n, -5.f, 5.f);
    fill_random_host(h_bias, 1024, -1.f, 1.f);

    for (int i = 0; i < n; ++i) {
        float v = h_x[i] + h_bias[i % 1024];
        h_ref[i] = 0.5f * v * (1.f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    float *d_x, *d_b, *d_y1, *d_y2;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y1, bytes));
    CUDA_CHECK(cudaMalloc(&d_y2, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_bias, 1024 * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;
    CudaTimer timer;

    // v1: two kernels
    timer.start();
    add_bias<<<grid, block>>>(d_x, d_b, d_y1, n);
    CHECK_KERNEL_LAUNCH();
    gelu_inplace<<<grid, block>>>(d_y1, d_y2, n);
    CHECK_KERNEL_LAUNCH();
    float ms_two = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y2, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, n, 1e-5f)) {
        std::printf("two-kernel version FAILED\n");
        return 1;
    }

    // v2: fused
    timer.start();
    bias_gelu_fused<<<grid, block>>>(d_x, d_b, d_y2, n);
    CHECK_KERNEL_LAUNCH();
    float ms_fused = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y2, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, n, 1e-5f)) {
        std::printf("fused version FAILED\n");
        return 1;
    }

    std::printf("two kernels: %8.3f ms\n", ms_two);
    std::printf("fused      : %8.3f ms  (skips one full read+write of the intermediate)\n",
                ms_fused);
    std::printf("verification passed\n");

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_y1);
    cudaFree(d_y2);
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);
    return 0;
}
