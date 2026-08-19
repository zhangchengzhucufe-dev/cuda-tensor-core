// Benchmark against cuBLAS. Knowing how far your hand-written kernel is
// from the library is basic hygiene for performance work -- "what fraction
// of cuBLAS did you reach?" is a question you *will* get asked.
//
// Same data, two runners:
//   1. my register-tile kernel (a copy of the one in sgemm_register_tile.cu,
//      kept self-contained so this file stands alone)
//   2. cublasSgemm
// Prints GFLOPS for both plus the ratio, and cross-checks my result
// against cuBLAS's.
//
// cuBLAS is column-major: row-major C = A*B is the same numbers as
// column-major C' = B'*A', so you swap the operand order and the leading
// dimensions (classic trick, worth remembering).
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "error_check.h"

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s__ = (call);                                          \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(s__));                    \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_regtile(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float acc[TM][TN] = {};
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    for (int t = 0; t < K / BK; ++t) {
        int a_row = ty * 8 + tx / 2;
        int a_col = (tx % 2) * 4;
        float4 av = *reinterpret_cast<const float4*>(
            A + (row_base + a_row) * K + t * BK + a_col);
        As[a_row][a_col] = av.x;
        As[a_row][a_col + 1] = av.y;
        As[a_row][a_col + 2] = av.z;
        As[a_row][a_col + 3] = av.w;

        int tid = ty * 16 + tx;
        int b_row = tid / 32;
        int b_col = (tid % 32) * 4;
        float4 bv = *reinterpret_cast<const float4*>(
            B + (t * BK + b_row) * N + col_base + b_col);
        Bs[b_row][b_col] = bv.x;
        Bs[b_row][b_col + 1] = bv.y;
        Bs[b_row][b_col + 2] = bv.z;
        Bs[b_row][b_col + 3] = bv.w;
        __syncthreads();

        float a_frag[TM], b_frag[TN];
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
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(row_base + ty * TM + i) * N + col_base + tx * TN + j] = acc[i][j];
}

int main(int argc, char** argv) {
    std::srand(0);
    int M = 2048, N = 2048, K = 2048;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (M % BM || N % BN || K % BK) {
        std::printf("need M%%128==0, N%%128==0, K%%8==0\n");
        return 1;
    }
    std::printf("GEMM vs cuBLAS: M=%d N=%d K=%d\n", M, N, K);

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

    float *h_a = static_cast<float*>(std::malloc(a_bytes));
    float *h_b = static_cast<float*>(std::malloc(b_bytes));
    fill_random_host(h_a, static_cast<size_t>(M) * K, -1.f, 1.f);
    fill_random_host(h_b, static_cast<size_t>(K) * N, -1.f, 1.f);

    float *d_a, *d_b, *d_c, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CudaTimer timer;
    double gflops = 2.0 * M * N * K / 1e9;

    // ---- cuBLAS (best of several, lets the library warm up its clocks) ----
    const float one = 1.f, zero = 0.f;
    float ms_cublas = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        // row-major C = A*B  ==  column-major C' = B'*A': B goes first,
        // leading dims are N and K
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                                 d_b, N, d_a, K, &zero, d_ref, N));
        ms_cublas = fminf(ms_cublas, timer.stop());
    }

    // ---- my register-tile kernel ----
    dim3 block(16, 16);
    dim3 grid(N / BN, M / BM);
    float ms_mine = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        sgemm_regtile<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        CHECK_KERNEL_LAUNCH();
        ms_mine = fminf(ms_mine, timer.stop());
    }

    std::printf("mine (register-tile): %8.3f ms, %8.1f GFLOPS\n", ms_mine,
                gflops / (ms_mine / 1e3));
    std::printf("cuBLAS              : %8.3f ms, %8.1f GFLOPS\n", ms_cublas,
                gflops / (ms_cublas / 1e3));
    std::printf("reached %.1f%% of cuBLAS\n", ms_cublas / ms_mine * 100.f);

    // cross-check mine against cuBLAS
    size_t c_elems = static_cast<size_t>(M) * N;
    float *h_c = static_cast<float*>(std::malloc(c_bytes));
    float *h_ref = static_cast<float*>(std::malloc(c_bytes));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, c_bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, c_elems, 1e-4f)) {
        std::printf("results disagree with cuBLAS!\n");
        return 1;
    }
    std::printf("matches cuBLAS, verification passed\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ref);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
