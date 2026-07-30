// Parallel reduction (sum). Same problem three times, each version fixes
// something the previous one does badly. The speedup from first to last
// is the whole lesson.
//
//   reduce_divergent  : the textbook first attempt -- adjacent pairs,
//                       if-condition causes divergence inside warps
//   reduce_interleaved: sequential addressing, active threads stay packed
//                       together, no divergence
//   reduce_warp_shfl  : each thread serially accumulates UNROLL elements
//                       first (less shared memory traffic), then warp-level
//                       reduction via __shfl_down_sync (never touches shared
//                       memory within a warp)
//
// All three do the same number of FLOPs. The gap is purely memory pattern
// and branching, which is kind of the point.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define BLOCK 256

// V1: adjacent pairs. At stride s, half the threads in a warp are active
// and they're scattered -- the modulo condition sends lanes both ways.
__global__ void reduce_divergent(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// V2: interleaved. Add top half to bottom half; the active threads are
// threads 0..stride-1, all packed. Fewer bank conflicts too.
__global__ void reduce_interleaved(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// V3: two levels of improvement.
//   a) each thread accumulates UNROLL elements in a register first. The
//      global reads stay coalesced, and the data shrinks UNROLLx before it
//      ever hits shared memory
//   b) within a block, reduce inside each warp with __shfl_down_sync (warp
//      is implicitly synchronized, shuffle swaps registers directly), only
//      one value per warp goes to shared memory, then warp 0 finishes up
#define UNROLL 8

__global__ void reduce_warp_shfl(const float* in, float* out, int n) {
    __shared__ float warp_sums[BLOCK / 32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * UNROLL) + tid;

    // (a) serial accumulate. Striding by blockDim.x keeps warp accesses merged
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        int idx = gid + k * blockDim.x;
        if (idx < n) sum += in[idx];
    }

    // (b) warp reduction: 5 shuffle rounds and lane 0 holds the warp's sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid % 32 == 0) warp_sums[tid / 32] = sum;
    __syncthreads();

    // First warp reduces the per-warp partials
    if (tid < 32) {
        float v = (tid < BLOCK / 32) ? warp_sums[tid] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) out[blockIdx.x] = v;
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 24;
    if (argc > 1) n = std::atoi(argv[1]);

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float* h_in = static_cast<float*>(std::malloc(bytes));
    fill_random_host(h_in, n, -1.f, 1.f);

    // CPU reference in double -- float sums have real rounding error
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += h_in[i];

    float* d_in;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    CudaTimer timer;
    float ms[3];
    const char* names[3] = {"divergent   ", "interleaved ", "warp_shfl   "};

    for (int ver = 0; ver < 3; ++ver) {
        // One partial sum per block; the rest is small enough to finish
        // serially on the CPU, no point writing a second kernel for a demo
        int grid = (n + BLOCK - 1) / BLOCK;
        if (ver == 2) grid = (n + BLOCK * UNROLL - 1) / (BLOCK * UNROLL);

        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, grid * sizeof(float)));

        timer.start();
        if (ver == 0)
            reduce_divergent<<<grid, BLOCK>>>(d_in, d_out, n);
        else if (ver == 1)
            reduce_interleaved<<<grid, BLOCK>>>(d_in, d_out, n);
        else
            reduce_warp_shfl<<<grid, BLOCK>>>(d_in, d_out, n);
        CHECK_KERNEL_LAUNCH();
        ms[ver] = timer.stop();

        float* h_part = static_cast<float*>(std::malloc(grid * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(h_part, d_out, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float total = 0.f;
        for (int i = 0; i < grid; ++i) total += h_part[i];


        double rel_err = std::fabs(total - ref) / std::fabs(ref);
        std::printf("%s: %8.3f ms, %7.1f GB/s, rel err %.2e\n", names[ver], ms[ver],
                    bytes / 1e9 / (ms[ver] / 1e3f), rel_err);
        if (rel_err > 1e-4) {
            std::printf("  -> error too large, FAILED\n");
            return 1;
        }
        std::free(h_part);
        cudaFree(d_out);
    }
    std::printf("reference (double): %.2f\n", ref);
    std::printf("all three passed -- check how much faster the last one is\n");

    cudaFree(d_in);
    std::free(h_in);
    return 0;
}
