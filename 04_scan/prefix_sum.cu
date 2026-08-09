// Prefix sum (scan). out[i] = in[0] + ... + in[i], the inclusive flavor.
//
// The naive parallelization does O(n log n) total work (Hillis-Steele).
// This implements the work-efficient Brent-Kung version: O(n) work, which
// is the best a parallel scan can do.
//
// For arrays bigger than one block, the standard three-pass structure:
//   1. each block Blelloch-scans its own 2*BLOCK slice and writes its
//      total out as a "block sum"
//   2. scan the block sums. There are only a few hundred of them, so
//      doing it on the CPU keeps the code short. If you want it fully
//      on the GPU, just feed the block sums back into scan_block and recurse
//   3. every block adds the sum of all *preceding* block sums to its elements
//
// Why 2*BLOCK per block: Blelloch's up/down sweep wants a power-of-two
// length, and with 512 threads handling 1024 elements each thread starts
// out owning exactly 2.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define BLOCK 512

// Work-efficient in-block scan (Blelloch 1990).
// Scans d_in[start .. start+2*BLOCK), writes inclusive scan to d_out,
// and leaves this slice's total in d_block_sums[blockIdx.x].
__global__ void scan_block(const float* d_in, float* d_out,
                           float* d_block_sums, int n) {
    __shared__ float temp[2 * BLOCK];

    int tid = threadIdx.x;
    long long start = static_cast<long long>(blockIdx.x) * (2 * BLOCK);

    // each thread brings in two elements
    long long i0 = start + 2 * tid;
    long long i1 = i0 + 1;
    temp[2 * tid] = (i0 < n) ? d_in[i0] : 0.f;
    temp[2 * tid + 1] = (i1 < n) ? d_in[i1] : 0.f;
    __syncthreads();

    // up-sweep (reduction): partial sums bottom-up
    int stride = 1;
    for (int d = 2 * BLOCK; d > 1; d >>= 1) {
        __syncthreads();
        if (tid < d / 2) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        stride <<= 1;
    }

    // stash the total (last element), zero it out, then down-sweep
    if (tid == 0) {
        if (d_block_sums) d_block_sums[blockIdx.x] = temp[2 * BLOCK - 1];
        temp[2 * BLOCK - 1] = 0.f;
    }

    // down-sweep: turn the partial sums into an exclusive scan
    for (int d = 1; d < 2 * BLOCK; d <<= 1) {
        stride >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = stride * (2 * tid + 1) - 1;
            int bi = stride * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // temp now holds the exclusive scan (temp[0] == 0); add the element
    // back to itself to make it inclusive
    if (i0 < n) d_out[i0] = temp[2 * tid] + d_in[i0];
    if (i1 < n) d_out[i1] = temp[2 * tid + 1] + d_in[i1];
}

// Pass 3: add each block's prefix-of-block-sums offset
__global__ void scan_add_offsets(float* d_out, const float* d_block_sums, int n) {
    long long start = static_cast<long long>(blockIdx.x) * (2 * BLOCK);
    float offset = d_block_sums[blockIdx.x];
    long long i = start + 2 * threadIdx.x;
    if (i < n) d_out[i] += offset;
    if (i + 1 < n) d_out[i + 1] += offset;
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 20;
    if (argc > 1) n = std::atoi(argv[1]);

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h_in = static_cast<float*>(std::malloc(bytes));
    float* h_out = static_cast<float*>(std::malloc(bytes));
    fill_random_host(h_in, n, 0.f, 1.f);

    // CPU reference in double so float rounding doesn't muddy the check
    double* h_ref = static_cast<double*>(std::malloc(n * sizeof(double)));
    double run = 0.0;
    for (int i = 0; i < n; ++i) {
        run += h_in[i];
        h_ref[i] = run;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int num_blocks = (n + 2 * BLOCK - 1) / (2 * BLOCK);
    float* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));

    CudaTimer timer;
    timer.start();
    // pass 1: per-block scan + export the block sums
    scan_block<<<num_blocks, BLOCK>>>(d_in, d_out, d_block_sums, n);
    CHECK_KERNEL_LAUNCH();

    // pass 2: the block sums are few (512 here), scanning them on the CPU
    // is the clearest. Watch out: this must be an EXCLUSIVE scan -- block b
    // needs the sum of blocks *before* b, and block 0 gets 0. I got this
    // wrong once and block 0's output was off by its own sum. Accumulate in
    // double while at it.
    float* h_block_sums = static_cast<float*>(std::malloc(num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double run_sums = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        double s = h_block_sums[i];
        h_block_sums[i] = static_cast<float>(run_sums);
        run_sums += s;
    }
    CUDA_CHECK(cudaMemcpy(d_block_sums, h_block_sums, num_blocks * sizeof(float),
                          cudaMemcpyHostToDevice));

    // pass 3: add the offsets back
    scan_add_offsets<<<num_blocks, BLOCK>>>(d_out, d_block_sums, n);
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    std::printf("GPU scan: %d elements, %.3f ms\n", n, ms);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        double limit = 1e-4 * (1.0 + std::fabs(h_ref[i]));
        if (std::fabs(h_out[i] - h_ref[i]) > limit) {
            std::fprintf(stderr, "FAILED at i=%d: got %f, expect %f\n", i,
                         h_out[i], h_ref[i]);
            return 1;
        }
    }
    std::printf("verification passed (Brent-Kung work-efficient scan, three passes)\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_block_sums);
    std::free(h_in);
    std::free(h_out);
    std::free(h_ref);
    std::free(h_block_sums);
    return 0;
}
