// Vector add. The "hello world" of CUDA, but with the stuff you actually
// need to make it a useful exercise:
//   1. a CPU reference + result check
//   2. cudaEvent timing of just the kernel, with effective bandwidth in GB/s
//
// The thing to internalize here: C[i] = A[i] + B[i] does 1 FLOP per
// 12 bytes of traffic (8 read + 4 write). This is a memory-bound kernel,
// so GB/s is the number to watch, not FLOPS.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // The global thread index. First line of basically every CUDA kernel.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check: if n isn't a multiple of blockDim, the last block
    // would run off the end without this.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 24;  // ~16M elements, 64MB per array
    if (argc > 1) n = std::atoi(argv[1]);

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::printf("n = %d (%.1f MB per array)\n", n, bytes / 1e6);

    float *h_a = static_cast<float*>(std::malloc(bytes));
    float *h_b = static_cast<float*>(std::malloc(bytes));
    float *h_c = static_cast<float*>(std::malloc(bytes));
    float *h_ref = static_cast<float*>(std::malloc(bytes));
    fill_random_host(h_a, n, -1.f, 1.f);
    fill_random_host(h_b, n, -1.f, 1.f);

    vector_add_cpu(h_a, h_b, h_ref, n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 256 threads per block is a fine default (8 warps, easy to schedule)
    int block = 256;
    int grid = (n + block - 1) / block;
    std::printf("launch: grid=%d, block=%d\n", grid, block);

    // Timing wraps only the kernel, not the H2D/D2H copies
    CudaTimer timer;
    timer.start();
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();

    // Effective bandwidth = (2 reads + 1 write) * bytes / time
    float gb = 3.f * bytes / 1e9f;
    std::printf("kernel time: %.3f ms, effective bandwidth: %.1f GB/s\n", ms,
                gb / (ms / 1e3f));

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, n, 1e-6f)) {
        std::printf("verification FAILED\n");
        return 1;
    }
    std::printf("verification passed\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
