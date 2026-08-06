// SGEMM v3: tensor cores via the WMMA API.
// A tensor core does a 16x16x16 matrix multiply-accumulate in one
// instruction, FP16 in / FP32 accumulate. WMMA (mma.h) is the friendliest
// way to touch tensor cores as a learner -- no PTX, no CUTLASS. The
// structure is basically tiled GEMM with the scalar loop replaced by
// fragment operations:
//
//   tiled GEMM : scalar FMA, one thread owns one element of C
//   WMMA       : fragments (register-resident matrix tiles), one warp owns
//                a 16x16 tile of C
//
// On precision: inputs get converted to half (~3 decimal digits) but the
// accumulation happens in FP32, so the result is much better than pure
// FP16 and fine for DL workloads.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>
#include "error_check.h"

using namespace nvcuda;

// One warp per block, one 16x16 tile of C per warp. Single warp -- the
// point is to understand fragments, not to squeeze the hardware.
#define WM 16
#define WN 16
#define WK 16

__global__ void sgemm_wmma(const half* A, const half* B, float* C,
                           int M, int N, int K) {
    // fragment: how tensor cores see a matrix tile. Lives in registers,
    // the physical layout is deliberately invisible to you
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;

    wmma::fill_fragment(c_frag, 0.f);

    // top-left corner of this warp's C tile
    int m_base = blockIdx.y * WM;
    int n_base = blockIdx.x * WN;

    for (int k = 0; k < K; k += WK) {
        const half* a_tile = A + static_cast<size_t>(m_base) * K + k;      // MxK row-major
        const half* b_tile = B + static_cast<size_t>(k) * N + n_base;      // KxN row-major
        wmma::load_matrix_sync(a_frag, a_tile, K);   // ldm = actual row stride
        wmma::load_matrix_sync(b_frag, b_tile, N);
        // the whole point: one instruction, 16x16x16 MMA
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_tile = C + static_cast<size_t>(m_base) * N + n_base;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
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
    // load_matrix_sync needs all dims to be multiples of 16 and 32B-aligned
    int M = 1024, N = 1024, K = 1024;
    if (argc > 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (M % WM || N % WN || K % WK) {
        std::printf("M, N, K must all be multiples of 16 in this example\n");
        return 1;
    }
    std::printf("SGEMM wmma (FP16 in / FP32 accum): M=%d N=%d K=%d\n", M, N, K);

    // generate float inputs on the host, then convert to half for the GPU
    size_t a_elems = static_cast<size_t>(M) * K;
    size_t b_elems = static_cast<size_t>(K) * N;
    float* h_a = static_cast<float*>(std::malloc(a_elems * sizeof(float)));
    float* h_b = static_cast<float*>(std::malloc(b_elems * sizeof(float)));
    fill_random_host(h_a, a_elems, -1.f, 1.f);
    fill_random_host(h_b, b_elems, -1.f, 1.f);

    half* h_ah = static_cast<half*>(std::malloc(a_elems * sizeof(half)));
    half* h_bh = static_cast<half*>(std::malloc(b_elems * sizeof(half)));
    for (size_t i = 0; i < a_elems; ++i) h_ah[i] = __float2half(h_a[i]);
    for (size_t i = 0; i < b_elems; ++i) h_bh[i] = __float2half(h_b[i]);

    half *d_a, *d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_ah, a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_bh, b_elems * sizeof(half), cudaMemcpyHostToDevice));

    // grid is just the number of 16x16 tiles, one warp per block
    dim3 block(32, 1);
    dim3 grid(N / WN, M / WM);

    CudaTimer timer;
    timer.start();
    sgemm_wmma<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
    float* h_c = static_cast<float*>(std::malloc(c_bytes));
    CUDA_CHECK(cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost));

    double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
    std::printf("kernel: %.3f ms, %.2f TFLOPS\n", ms, tflops);

    // Verify against a reference computed on the *half-rounded* inputs --
    // the half rounding is where the error comes from, that's expected,
    // not a bug. Tolerance is relaxed accordingly.
    float* h_af = static_cast<float*>(std::malloc(a_elems * sizeof(float)));
    float* h_bf = static_cast<float*>(std::malloc(b_elems * sizeof(float)));
    for (size_t i = 0; i < a_elems; ++i) h_af[i] = __half2float(h_ah[i]);
    for (size_t i = 0; i < b_elems; ++i) h_bf[i] = __half2float(h_bh[i]);

    double* h_ref = static_cast<double*>(
        std::malloc(static_cast<size_t>(M) * N * sizeof(double)));
    sgemm_cpu(h_af, h_bf, h_ref, M, N, K);
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
        double limit = 5e-2 * (1.0 + std::fabs(h_ref[i]));
        if (std::fabs(h_c[i] - h_ref[i]) > limit) {
            std::fprintf(stderr, "FAILED at %zu: got %f, expect %f\n", i, h_c[i],
                         h_ref[i]);
            return 1;
        }
    }
    std::printf("verification passed. FP32 accumulation keeps the FP16 input error tiny\n");
    std::free(h_ref);
    std::free(h_af);
    std::free(h_bf);
    std::free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_ah);
    std::free(h_bh);
    return 0;
}
