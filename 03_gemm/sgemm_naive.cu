// SGEMM v1: naive. One thread computes one element of C.
// C = A(M x K) * B(K x N), row-major.
//
// This lands at a few percent of peak FLOPS. Why: every single FMA
// re-reads an element of A's row and B's column from global memory.
// Zero data reuse, terrible arithmetic intensity. The B reads aren't
// even coalesced (adjacent threads differ by k, so addresses differ by N
// floats). sgemm_tiled fixes the reuse with shared memory.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// CPU reference with double accumulation, used as the ground truth
void sgemm_cpu(const float* A, const float* B, double* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                       B[static_cast<size_t>(k) * N + j];
            }
            C[static_cast<size_t>(i) * N + j] = acc;
        }
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int M = 1024, N = 1024, K = 1024;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    std::printf("SGEMM naive: M=%d N=%d K=%d\n", M, N, K);

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

    float *h_a = static_cast<float*>(std::malloc(a_bytes));
    float *h_b = static_cast<float*>(std::malloc(b_bytes));
    float *h_c = static_cast<float*>(std::malloc(c_bytes));
    fill_random_host(h_a, static_cast<size_t>(M) * K, -1.f, 1.f);
    fill_random_host(h_b, static_cast<size_t>(K) * N, -1.f, 1.f);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    // 32 in x so consecutive threads handle consecutive columns -> coalesced
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    CudaTimer timer;
    timer.start();
    sgemm_naive<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));

    double gflops = 2.0 * M * N * K / (ms / 1e3) / 1e9;
    std::printf("kernel: %.3f ms, %.1f GFLOPS\n", ms, gflops);

    // CPU GEMM is O(N^3); a couple of seconds at 1024 is normal
    double* h_ref = static_cast<double*>(std::malloc(c_bytes * sizeof(double)));
    sgemm_cpu(h_a, h_b, h_ref, M, N, K);
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        double limit = 1e-3 * (1.0 + std::fabs(h_ref[i]));
        if (std::fabs(h_c[i] - h_ref[i]) > limit) {
            std::fprintf(stderr, "FAILED at %zu: got %f, expect %f\n", i, h_c[i],
                         h_ref[i]);
            return 1;
        }
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
