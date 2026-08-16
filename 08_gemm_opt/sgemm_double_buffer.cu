// SGEMM v5: double buffering with cp.async.
// Next rung on the ladder after register tiling. In the single-buffer
// version every K-step has a dead phase: load tiles into shared memory,
// sync, compute, sync, repeat. The memory pipe sits idle while compute
// runs and vice versa.
//
// Double buffering hides the load behind the compute: shared memory has
// two sets of tiles, and while the block crunches tile t it's already
// pulling tile t+1 into the other buffer. cp.async (the async copy path,
// sm_80+) does the global->shared copy straight from registers without
// stalling the warp.
//
// Layout is the same 128x128x8 recipe as sgemm_register_tile.cu, just with
// two buffers and the load/compute phases interleaved.
//
// Honest result on this card (RTX 3060 Laptop, 2048^3): ~5.4 TFLOPS vs
// ~4.7 for the single-buffer version, +15% -- and the interesting part is
// why it's *only* +15%, not the 2x the blog posts promise:
//   - with BK=8, one pipeline stage ahead is only ~512 FMAs per thread,
//     not enough compute to cover a full global memory round trip
//   - registers cap this kernel at 2 blocks/SM (~123 regs x 256 threads =
//     half the register file), so there's little cross-block overlap to
//     lean on -- which is also why hiding the loads inside the block helps
//     at all
//   - the kernel is close to FMA-pipe-bound now (SM SOL 47% -> 57%,
//     docs/gemm_deep_dive.md); prefetch can't feed a pipe that's already
//     near-saturated
// Making this pay off needs deeper pipelining: 3-4 stages, or a wider K
// tile (BK=16+) so each wait covers more compute. That's exactly what
// CUTLASS's multistage pipelines do. Kept at BK=8 here because the point
// of the file is the cp.async mechanics, not the last 10%.
#include <cstdio>
#include <cstdlib>
#include <cuda_pipeline.h>  // __pipeline_memcpy_async / commit / wait_prior
#include "error_check.h"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_double_buffer(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    // double the tiles: [slot][...]. Bs rows are padded to BN+4: cp.async
    // needs 16B-aligned shared dst, so the row stride (132 floats = 528B)
    // must stay a multiple of 4 floats -- a plain +1 pad would misalign
    // every other row and cp.async faults out
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float acc[TM][TN] = {};

    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;
    int num_tiles = K / BK;

    // issues the global->shared copies for tile t into slot buf via cp.async,
    // one float4 per thread for each of A and B (same split as the
    // single-buffer version)
    auto issue_tile = [&](int t, int buf) {
        int a_row = ty * 8 + tx / 2;
        int a_col = (tx % 2) * 4;
        __pipeline_memcpy_async(&As[buf][a_row][a_col],
                                A + (row_base + a_row) * K + t * BK + a_col, 16);
        int tid = ty * 16 + tx;
        int b_row = tid / 32;
        int b_col = (tid % 32) * 4;
        __pipeline_memcpy_async(&Bs[buf][b_row][b_col],
                                B + (t * BK + b_row) * N + col_base + b_col, 16);
    };

    // prologue: start pulling tile 0
    issue_tile(0, 0);
    __pipeline_commit();

    for (int t = 0; t < num_tiles; ++t) {
        int buf = t & 1;
        // queue up the next tile while waiting on the current one
        if (t + 1 < num_tiles) {
            issue_tile(t + 1, buf ^ 1);
            __pipeline_commit();
            __pipeline_wait_prior(1);  // everything except the newest group
        } else {
            __pipeline_wait_prior(0);  // last tile, nothing else in flight
        }
        __syncthreads();  // tile t visible to the whole block now

        float a_frag[TM];
        float b_frag[TN];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[buf][ty * TM + i][k];
            // two float4 reads instead of 8 scalars: the row stride is a
            // multiple of 4 floats so these are 16B aligned, and 128-bit
            // shared reads don't bank conflict the way scalar strides do
            *reinterpret_cast<float4*>(&b_frag[0]) =
                *reinterpret_cast<float4*>(&Bs[buf][k][tx * TN]);
            *reinterpret_cast<float4*>(&b_frag[4]) =
                *reinterpret_cast<float4*>(&Bs[buf][k][tx * TN + 4]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
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
    if (M % BM || N % BN || K % BK) {
        std::printf("need M%%128==0, N%%128==0, K%%8==0\n");
        return 1;
    }
    std::printf("SGEMM double-buffer (cp.async): M=%d N=%d K=%d\n", M, N, K);

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
    float ms = 1e30f;
    for (int rep = 0; rep < 2; ++rep) {
        timer.start();
        sgemm_double_buffer<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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
    std::printf("verification passed (compare with sgemm_register_tile in this dir)\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);
    return 0;
}
