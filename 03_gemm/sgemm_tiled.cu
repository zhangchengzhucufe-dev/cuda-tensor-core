// SGEMM v2: shared memory tiling.
// Load a 32x32 tile of A and of B into shared memory, and every thread in
// the block does its dot product against the tiles. Each A/B element gets
// reused 32 times, so global memory traffic drops by a factor of TILE.
//
// Basically every fast GEMM (cuBLAS, CUTLASS) is built on this idea --
// they just stack register tiling, vectorized loads, double buffering etc.
// on top of it. That stacking happens in 08_gemm_opt.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define TILE 32

__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float acc = 0.f;
    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // Cooperative load: every thread grabs one element of each tile.
        // Out-of-range positions get 0 when K isn't a multiple of TILE
        // (0 * anything doesn't hurt the accumulate)
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();  // tile has to be fully loaded

        // Dot product within the tile: all reads hit shared memory
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();  // everyone done with the tile before it gets overwritten
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

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
    std::printf("SGEMM tiled: M=%d N=%d K=%d\n", M, N, K);

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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    CudaTimer timer;
    timer.start();
    sgemm_tiled<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));

    double gflops = 2.0 * M * N * K / (ms / 1e3) / 1e9;
    std::printf("kernel: %.3f ms, %.1f GFLOPS\n", ms, gflops);

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
    std::printf("verification passed (compare GFLOPS against the naive version)\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
