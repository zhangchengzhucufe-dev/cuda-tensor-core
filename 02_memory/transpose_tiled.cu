// Matrix transpose: THE example for coalesced memory access + shared memory.
//
// In row-major layout a transpose always has one side with strided access:
//   naive: reads in[x*n+y] with stride n (uncoalesced), writes coalesced
//   tiled: both sides coalesced. Read into shared memory the fast way,
//          transpose inside the tile, write out the fast way.
//
// The shared tile is declared [TILE][TILE+1] -- the +1 kills the bank
// conflict where an entire column lands on the same bank in a 32x32
// float tile. Same trick shows up everywhere, worth memorizing.
//
// Metric is GB/s again: each element gets read once and written once.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define TILE 32

__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row
    if (x < n && y < n) {
        // Writes out[y*n+x] are coalesced (x varies fastest across threads)
        // but reads in[x*n+y] stride by n. Completely uncoalesced. Slow.
        out[y * n + x] = in[x * n + y];
    }
}

__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];  // +1 to avoid bank conflicts

    // Step 1: read the input into shared memory, coalesced along the row
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();  // have to wait for the whole tile

    // Step 2: swap the roles of blockIdx.x/y so the *write* is also
    // coalesced (consecutive threads write consecutive addresses)
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < n && y < n) {
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_cpu(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out[static_cast<size_t>(j) * n + i] = in[static_cast<size_t>(i) * n + j];
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 4096;  // 4096x4096, 64MB
    if (argc > 1) n = std::atoi(argv[1]);

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    std::printf("matrix %d x %d (%.1f MB)\n", n, n, bytes / 1e6);

    float *h_in = static_cast<float*>(std::malloc(bytes));
    float *h_out = static_cast<float*>(std::malloc(bytes));
    float *h_ref = static_cast<float*>(std::malloc(bytes));
    fill_random_host(h_in, static_cast<size_t>(n) * n);
    transpose_cpu(h_in, h_ref, n);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    CudaTimer timer;

    timer.start();
    transpose_naive<<<grid, block>>>(d_in, d_out, n);
    CHECK_KERNEL_LAUNCH();
    float ms_naive = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_out, h_ref, static_cast<size_t>(n) * n, 0.f)) {
        std::printf("naive transpose FAILED\n");
        return 1;
    }

    timer.start();
    transpose_tiled<<<grid, block>>>(d_in, d_out, n);
    CHECK_KERNEL_LAUNCH();
    float ms_tiled = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    if (!compare_close(h_out, h_ref, static_cast<size_t>(n) * n, 0.f)) {
        std::printf("tiled transpose FAILED\n");
        return 1;
    }

    float gb = 2.f * bytes / 1e9f;  // read once + write once
    std::printf("naive : %8.3f ms, %7.1f GB/s\n", ms_naive, gb / (ms_naive / 1e3f));
    std::printf("tiled : %8.3f ms, %7.1f GB/s  (shared memory, no bank conflicts)\n",
                ms_tiled, gb / (ms_tiled / 1e3f));

    cudaFree(d_in);
    cudaFree(d_out);
    std::free(h_in);
    std::free(h_out);
    std::free(h_ref);
    return 0;
}
