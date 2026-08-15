// SGEMM v4: 2D register tiling + vectorized loads.
// This is the step that takes you from "textbook tiled GEMM" toward
// cuBLAS territory -- and it's the biggest single win in the GEMM ladder
// (naive -> tiled -> register tile -> double buffer -> ...).
//
// The idea: before, each thread computed ONE element of C, so every shared
// memory read fed exactly one FMA. Now each thread computes 8x8 = 64
// outputs:
//   - one shared memory read feeds 8 FMAs, arithmetic intensity jumps
//   - the intermediates live in 64 registers and never touch memory
//   - global loads go through float4, 4 floats per instruction
//
// Layout (the classic 128x128x8 recipe, same one PMPP and the GEMM blogs use):
//   block: 16x16 = 256 threads, owns a 128x128 tile of C
//   per thread: an 8x8 output block, marching along K in steps of BK=8
//   shared: As[128][8], Bs[8][129] (+1 against bank conflicts, same trick
//   as the transpose example)
//
// Left on the table as exercises: float4 on the
// shared->register path, double buffering with async copies, swizzling.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_regtile(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN + 1];  // +1: threads read Bs 8 columns apart, keep the banks apart too

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    // this thread's 64-float accumulator, register-resident the whole time
    float acc[TM][TN] = {};

    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    int num_tiles = K / BK;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- cooperative load, all float4 ----
        // As: 128 rows x 8 cols = 2 float4 per row = 256 float4 = one per thread
        int a_row = ty * 8 + tx / 2;          // 0..127
        int a_col = (tx % 2) * 4;             // 0 or 4
        float4 av = *reinterpret_cast<const float4*>(
            A + (row_base + a_row) * K + t * BK + a_col);
        As[a_row][a_col + 0] = av.x;
        As[a_row][a_col + 1] = av.y;
        As[a_row][a_col + 2] = av.z;
        As[a_row][a_col + 3] = av.w;

        // Bs: 8 rows x 128 cols = 32 float4 per row = 256 float4 = one per thread
        int tid = ty * 16 + tx;
        int b_row = tid / 32;                 // 0..7
        int b_col = (tid % 32) * 4;           // 0,4,...,124
        float4 bv = *reinterpret_cast<const float4*>(
            B + (t * BK + b_row) * N + col_base + b_col);
        Bs[b_row][b_col + 0] = bv.x;
        Bs[b_row][b_col + 1] = bv.y;
        Bs[b_row][b_col + 2] = bv.z;
        Bs[b_row][b_col + 3] = bv.w;
        __syncthreads();

        // ---- compute: every shared memory element gets reused 8 times ----
        float a_frag[TM];
        float b_frag[TN];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[ty * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();  // everyone done before the next tile overwrites
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(row_base + ty * TM + i) * N + col_base + tx * TN + j] = acc[i][j];
        }
    }
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
    int M = 2048, N = 2048, K = 2048;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    // no boundary handling in the teaching version: sizes must be multiples
    // of the block dims (a real GEMM pads or branches, out of scope here)
    if (M % BM || N % BN || K % BK) {
        std::printf("need M%%128==0, N%%128==0, K%%8==0 (this kernel has no bounds checks)\n");
        return 1;
    }
    std::printf("SGEMM register-tile: M=%d N=%d K=%d (%dx%d outputs per thread)\n", M, N, K,
                TM, TN);

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

    dim3 block(16, 16);
    dim3 grid(N / BN, M / BM);

    CudaTimer timer;
    // run twice, keep the better one -- smooths out clock wobble
    float ms = 1e30f;
    for (int rep = 0; rep < 2; ++rep) {
        timer.start();
        sgemm_regtile<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        CHECK_KERNEL_LAUNCH();
        ms = fminf(ms, timer.stop());
    }
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
    std::printf("verification passed (compare GFLOPS with the tiled version in 03_gemm)\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
