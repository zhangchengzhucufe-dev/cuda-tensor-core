// Histogram: the canonical atomics example.
// Count 256 bins over a stream of random bytes. Two versions:
//
//   v1 (naive): every thread does atomicAdd straight into global memory.
//     Correct, but 16M elements fighting over 256 bins = constant conflicts.
//
//   v2 (privatization): each block builds its own private histogram in
//     shared memory first (shared atomics are much cheaper than global
//     ones), then merges once into global at the end. Global conflicts
//     drop from n to num_blocks * 256.
//
// The general lesson: lots of threads contending for few counters. So do
// the contention locally and merge once. This pattern shows up all over
// the place in real libraries.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define NUM_BINS 256

__global__ void histogram_naive(const unsigned char* data, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // atomicAdd makes the read-modify-write uninterruptible, returns old value
        atomicAdd(&hist[data[i]], 1);
    }
}

// Each thread serially handles COARSE elements: fewer atomics per thread,
// and the loop unrolls better. Cheap win.
#define COARSE 8

__global__ void histogram_privatized(const unsigned char* data, int* hist, int n) {
    __shared__ int local_hist[NUM_BINS];

    // cooperative zeroing of shared memory
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE;
    for (int k = 0; k < COARSE; ++k) {
        int i = start + k;
        if (i < n) {
            atomicAdd(&local_hist[data[i]], 1);  // conflicts only within this block
        }
    }
    __syncthreads();

    // merge back: just a handful of global atomics per block
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        atomicAdd(&hist[b], local_hist[b]);
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 24;  // 16M bytes
    if (argc > 1) n = std::atoi(argv[1]);
    std::printf("input: %d bytes\n", n);

    // random byte stream + CPU reference histogram
    unsigned char* h_data = static_cast<unsigned char*>(std::malloc(n));
    int ref_hist[NUM_BINS] = {0};
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<unsigned char>(std::rand() % NUM_BINS);
        ref_hist[h_data[i]]++;
    }

    unsigned char* d_data;
    int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    int h_hist[NUM_BINS];
    CudaTimer timer;
    int block = 256;

    // v1: naive
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
    int grid = (n + block - 1) / block;
    timer.start();
    histogram_naive<<<grid, block>>>(d_data, d_hist, n);
    CHECK_KERNEL_LAUNCH();
    float ms1 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < NUM_BINS; ++b) {
        if (h_hist[b] != ref_hist[b]) {
            std::fprintf(stderr, "naive FAILED: bin %d got %d expect %d\n",
                         b, h_hist[b], ref_hist[b]);
            return 1;
        }
    }
    std::printf("naive     : %8.3f ms  (%.1f MB/s)\n", ms1, n / 1e6 / (ms1 / 1e3));

    // v2: shared-memory privatization + thread coarsening
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
    grid = (n + block * COARSE - 1) / (block * COARSE);
    timer.start();
    histogram_privatized<<<grid, block>>>(d_data, d_hist, n);
    CHECK_KERNEL_LAUNCH();
    float ms2 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < NUM_BINS; ++b) {
        if (h_hist[b] != ref_hist[b]) {
            std::fprintf(stderr, "privatized FAILED: bin %d got %d expect %d\n",
                         b, h_hist[b], ref_hist[b]);
            return 1;
        }
    }
    std::printf("privatized: %8.3f ms  (%.1f MB/s)\n", ms2, n / 1e6 / (ms2 / 1e3));
    std::printf("both passed\n");

    cudaFree(d_data);
    cudaFree(d_hist);
    std::free(h_data);
    return 0;
}
