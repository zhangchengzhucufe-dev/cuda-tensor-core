// The full GEMM fight, one file, all verified against each other:
//   1. my register-tile kernel (SIMT, copied from sgemm_vs_cublas.cu)
//   2. CUTLASS stock SIMT config (128x128x8, same shape as mine)
//   3. CUTLASS TF32 tensor cores (float in/out, 10-bit mantissa in the MMA)
//   4. CUTLASS FP16 tensor cores, FP32 accumulate, FP16 out (the standard
//      inference config)
//   5. cublasSgemm (FP32, the "library" baseline everyone quotes)
//   6. cublasGemmEx FP16 in/out with FP32 compute (the library's tensor-core
//      ceiling on this card)
//
// Why the tensor-core rungs belong in the same file as the SIMT ladder:
// "SIMT is FMA-pipe-bound at 67% of peak" is only half the story. Tensor
// cores collapse the cost of the FFMA itself, which moves the bottleneck
// back to feeding the machine -- the numbers below make that visible.
//
// Tolerances are per-variant: TF32 truncates inputs to 10-bit
// mantissa (fp32 accumulate), FP16 rounds inputs AND output to fp16 --
// both are "wrong" vs the fp32 reference at their own scale, so each gets
// a tolerance that's tight for it rather than one number that's loose for
// SIMT and unfair on TF32.
//
// CUTLASS is header-only; set CUTLASS_DIR (default ~/cutlass) when building.
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "error_check.h"

#define CUTLASS_CHECK(call)                                                   \
    do {                                                                      \
        cutlass::Status s__ = (call);                                         \
        if (s__ != cutlass::Status::kSuccess) {                               \
            std::fprintf(stderr, "CUTLASS error at %s:%d: %s\n", __FILE__,    \
                         __LINE__, cutlassGetStatusString(s__));              \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s__ = (call);                                          \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(s__));                    \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Stock sm_80 SIMT config: 128x128x8 threadblock tiles, fp32 accumulate.
// Same tile shape as my hand-written kernel, so the ratio is apples to apples.
using GemmSimt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80>;

// TF32 tensor cores: same float tensors as above, but the MMA reads a
// 10-bit-mantissa snapshot of each input. Ampere's "free precision loss,
// double the tensor throughput" mode. Shapes are pinned explicitly -- the
// mma instruction is 16x8x8 and the defaults don't resolve to it on their
// own (compile dies with an incomplete Mma type). 3 stages x (A+B) 128x32
// float tiles = 96 KB smem, fits sm_86's 99 KB opt-in ceiling.
using GemmTf32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>>;

// FP16 in, FP32 accumulate, FP16 out: what inference engines actually run.
// Same story: explicit 16x8x16 HMMA shapes, 3 stages x 128x64 half tiles =
// 96 KB smem.
using GemmF16 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>>;

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

// half helpers: cutlass::half_t is bit-compatible with __half, so one pair
// of cast kernels serves both the CUTLASS fp16 path and cublasGemmEx.
__global__ void cast_f32_to_f16(const float* src, half* dst, size_t n) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void cast_f16_to_f32(const half* src, float* dst, size_t n) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
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
    std::printf("GEMM, the whole ladder: M=%d N=%d K=%d\n", M, N, K);

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
    size_t c_elems = static_cast<size_t>(M) * N;

    float* h_a = static_cast<float*>(std::malloc(a_bytes));
    float* h_b = static_cast<float*>(std::malloc(b_bytes));
    fill_random_host(h_a, static_cast<size_t>(M) * K, -1.f, 1.f);
    fill_random_host(h_b, static_cast<size_t>(K) * N, -1.f, 1.f);

    float *d_a, *d_b, *d_c, *d_ref;
    half *d_a16, *d_b16, *d_c16, *d_ref16;
    CUDA_CHECK(cudaMalloc(&d_a, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_a16, a_bytes / 2));
    CUDA_CHECK(cudaMalloc(&d_b16, b_bytes / 2));
    CUDA_CHECK(cudaMalloc(&d_c16, c_bytes / 2));
    CUDA_CHECK(cudaMalloc(&d_ref16, c_bytes / 2));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice));

    // one grid per buffer: A is MxK and B is KxN, sizing both off MxN
    // silently misses elements on non-square shapes (square hides it)
    int cast_a = static_cast<int>(((size_t)M * K + 255) / 256);
    int cast_b = static_cast<int>(((size_t)K * N + 255) / 256);
    int cast_c = static_cast<int>(((size_t)M * N + 255) / 256);
    cast_f32_to_f16<<<cast_a, 256>>>(d_a, d_a16, static_cast<size_t>(M) * K);
    cast_f32_to_f16<<<cast_b, 256>>>(d_b, d_b16, static_cast<size_t>(K) * N);
    CHECK_KERNEL_LAUNCH();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CudaTimer timer;
    double gflops = 2.0 * M * N * K / 1e9;
    const float one = 1.f, zero = 0.f;

    // ---- cuBLAS fp32: the reference everyone gets compared to ----
    float ms_cublas = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        // row-major C = A*B == column-major C' = B'*A': swap operand order
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                                 d_b, N, d_a, K, &zero, d_ref, N));
        ms_cublas = fminf(ms_cublas, timer.stop());
    }

    // ---- cuBLAS fp16 tensor cores (fp32 compute): the library's ceiling
    // on this card. Same column-major operand swap as above. ----
    float ms_cublas16 = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, d_b16,
            CUDA_R_16F, N, d_a16, CUDA_R_16F, K, &zero, d_ref16, CUDA_R_16F,
            N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        ms_cublas16 = fminf(ms_cublas16, timer.stop());
    }

    // ---- CUTLASS SIMT ----
    GemmSimt gemm_simt;
    CUTLASS_CHECK(gemm_simt.initialize({{M, N, K}, {d_a, K}, {d_b, N},
                                        {d_c, N}, {d_c, N}, {1.f, 0.f}}));
    float ms_simt = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        CUTLASS_CHECK(gemm_simt.run());
        ms_simt = fminf(ms_simt, timer.stop());
    }

    // ---- CUTLASS TF32 tensor cores ----
    GemmTf32 gemm_tf32;
    CUTLASS_CHECK(gemm_tf32.initialize({{M, N, K}, {d_a, K}, {d_b, N},
                                        {d_c, N}, {d_c, N}, {1.f, 0.f}}));
    float ms_tf32 = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        CUTLASS_CHECK(gemm_tf32.run());
        ms_tf32 = fminf(ms_tf32, timer.stop());
    }

    // ---- CUTLASS FP16 tensor cores ----
    // buffers are plain __half; cutlass wants cutlass::half_t*, same 16 bits
    GemmF16 gemm_f16;
    CUTLASS_CHECK(gemm_f16.initialize(
        {{M, N, K},
         {reinterpret_cast<cutlass::half_t*>(d_a16), K},
         {reinterpret_cast<cutlass::half_t*>(d_b16), N},
         {reinterpret_cast<cutlass::half_t*>(d_c16), N},
         {reinterpret_cast<cutlass::half_t*>(d_c16), N},
         {1.f, 0.f}}));
    float ms_f16 = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start();
        CUTLASS_CHECK(gemm_f16.run());
        ms_f16 = fminf(ms_f16, timer.stop());
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

    std::printf("mine (register-tile)  : %8.3f ms, %8.1f GFLOPS, %5.1f%% of cuBLAS\n",
                ms_mine, gflops / (ms_mine / 1e3), ms_cublas / ms_mine * 100.f);
    std::printf("CUTLASS (simt 128x128): %8.3f ms, %8.1f GFLOPS, %5.1f%% of cuBLAS\n",
                ms_simt, gflops / (ms_simt / 1e3), ms_cublas / ms_simt * 100.f);
    std::printf("CUTLASS (tf32 tensor) : %8.3f ms, %8.1f GFLOPS, %5.1f%% of cuBLAS\n",
                ms_tf32, gflops / (ms_tf32 / 1e3), ms_cublas / ms_tf32 * 100.f);
    std::printf("CUTLASS (fp16 tensor) : %8.3f ms, %8.1f GFLOPS, %5.1f%% of cuBLAS\n",
                ms_f16, gflops / (ms_f16 / 1e3), ms_cublas / ms_f16 * 100.f);
    std::printf("cuBLAS (sgemm fp32)   : %8.3f ms, %8.1f GFLOPS\n", ms_cublas,
                gflops / (ms_cublas / 1e3));
    std::printf("cuBLAS (fp16 tensor)  : %8.3f ms, %8.1f GFLOPS\n", ms_cublas16,
                gflops / (ms_cublas16 / 1e3));

    // ---- verification, each variant against the fp32 reference at its own
    // tolerance ----
    float *h_c = static_cast<float*>(std::malloc(c_bytes));
    float *h_ref = static_cast<float*>(std::malloc(c_bytes));
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, c_bytes, cudaMemcpyDeviceToHost));

    // mine + simt + tf32 all wrote d_c; run them again one at a time so the
    // buffer always holds exactly the variant under test
    sgemm_regtile<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, c_elems, 1e-4f)) {
        std::printf("mine disagrees with cuBLAS!\n");
        return 1;
    }
    CUTLASS_CHECK(gemm_simt.run());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, c_elems, 1e-4f)) {
        std::printf("CUTLASS simt disagrees with cuBLAS!\n");
        return 1;
    }
    CUTLASS_CHECK(gemm_tf32.run());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    // tolerance note: K=2048 rounds each input to 10-bit mantissa, and the
    // per-element error accumulates like sqrt(K) * 2^-11 ~= 2e-2 with the
    // cancellation in a random [-1,1] sum -- 5e-3 was measured too tight.
    if (!compare_close(h_c, h_ref, c_elems, 5e-2f)) {
        std::printf("CUTLASS tf32 disagrees with cuBLAS!\n");
        return 1;
    }

    // fp16 variants: same sqrt(K) logic, plus fp16 output rounding on top
    CUTLASS_CHECK(gemm_f16.run());
    cast_f16_to_f32<<<cast_c, 256>>>(d_c16, d_c, c_elems);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, c_elems, 5e-2f)) {
        std::printf("CUTLASS fp16 disagrees with cuBLAS!\n");
        return 1;
    }
    cast_f16_to_f32<<<cast_c, 256>>>(d_ref16, d_c, c_elems);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_c, h_ref, c_elems, 5e-2f)) {
        std::printf("cuBLAS fp16 disagrees with cuBLAS fp32 (does not happen,"
                    " investigate!)\n");
        return 1;
    }
    std::printf("all six match the fp32 reference at their own tolerance\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ref);
    cudaFree(d_a16);
    cudaFree(d_b16);
    cudaFree(d_c16);
    cudaFree(d_ref16);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
