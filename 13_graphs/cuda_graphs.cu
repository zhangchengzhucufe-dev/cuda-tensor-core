// CUDA Graphs: record a fixed sequence of kernels once, replay it many times.
//
// Launching kernels one at a time costs a few microseconds of CPU work per
// launch (arg packing, driver validation, scheduling). For tiny kernels
// that overhead dominates -- inference servers run hundreds of small ops
// per step and lose a big chunk of time to it. CUDA Graphs lets you capture
// a fixed pipeline with stream capture, instantiate it once, and replay
// with a single cudaGraphLaunch per frame. The driver also gets to see the
// whole graph, which helps scheduling.
//
// This demo builds a 3-kernel per-frame pipeline and runs 200 frames:
//   v1: launch 3 kernels individually, 200 times (600 launches)
//   v2: capture one frame into a graph, replay it 200 times
// The kernels are deliberately tiny so launch overhead shows up clearly.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define FRAMES 200
#define BLOCK 256

__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

__global__ void clamp_tanh(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(out[i]);
}

// Third stage: each block sums its 256 elements onto sums[bid].
// Done as a shared-memory tree reduction: float addition is
// order-dependent, so a plain `sums[bid] += out[i]` across 256 racing
// threads isn't deterministic and the two versions below would disagree.
__global__ void block_accumulate(const float* out, float* sums, int n) {
    __shared__ float buf[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[threadIdx.x] = (i < n) ? out[i] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] += buf[0];
}

int main() {
    std::srand(0);
    int n = 1 << 16;  // 64K elements: kernels are a few microseconds, so
                      // launch overhead is a big slice of the total
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    int grid = (n + BLOCK - 1) / BLOCK;

    float *h_x = static_cast<float*>(std::malloc(bytes));
    float *h_y = static_cast<float*>(std::malloc(bytes));
    fill_random_host(h_x, n, -1.f, 1.f);
    fill_random_host(h_y, n, -1.f, 1.f);

    float *d_x, *d_y, *d_out, *d_sums;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_sums, grid * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sums, 0, grid * sizeof(float)));

    CudaTimer timer;
    const float alpha = 0.5f;

    // ---- v1: individual launches, 200 frames x 3 kernels ----
    timer.start();
    for (int f = 0; f < FRAMES; ++f) {
        saxpy<<<grid, BLOCK>>>(alpha, d_x, d_y, d_out, n);
        clamp_tanh<<<grid, BLOCK>>>(d_out, n);
        block_accumulate<<<grid, BLOCK>>>(d_out, d_sums, n);
    }
    CHECK_KERNEL_LAUNCH();
    float ms_serial = timer.stop();
    size_t sums_bytes = static_cast<size_t>(grid) * sizeof(float);
    float* h_sums_a = static_cast<float*>(std::malloc(sums_bytes));
    CUDA_CHECK(cudaMemcpy(h_sums_a, d_sums, sums_bytes, cudaMemcpyDeviceToHost));

    // ---- v2: stream capture one frame -> graph -> replay 200 times ----
    CUDA_CHECK(cudaMemset(d_sums, 0, sums_bytes));
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    // no syncs or queries allowed during capture; the three kernels just
    // get recorded into the graph
    CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    saxpy<<<grid, BLOCK, 0, capture_stream>>>(alpha, d_x, d_y, d_out, n);
    clamp_tanh<<<grid, BLOCK, 0, capture_stream>>>(d_out, n);
    block_accumulate<<<grid, BLOCK, 0, capture_stream>>>(d_out, d_sums, n);
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    timer.start();
    for (int f = 0; f < FRAMES; ++f) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, nullptr));
    }
    float ms_graph = timer.stop();

    float* h_sums_b = static_cast<float*>(std::malloc(sums_bytes));
    CUDA_CHECK(cudaMemcpy(h_sums_b, d_sums, sums_bytes, cudaMemcpyDeviceToHost));

    // both versions run the same deterministic computation, results must
    // match bit for bit
    for (int i = 0; i < grid; ++i) {
        if (h_sums_a[i] != h_sums_b[i]) {
            std::fprintf(stderr, "FAILED at %d: %f vs %f\n", i, h_sums_a[i],
                         h_sums_b[i]);
            return 1;
        }
    }

    std::printf("%d frames x 3 kernels, n=%d (tiny kernels, launch cost dominates)\n", FRAMES, n);
    std::printf("individual launches: %8.3f ms\n", ms_serial);
    std::printf("graph replay       : %8.3f ms  (saved %.0f%% of the total, almost all launch overhead)\n",
                ms_graph, 100.f * (1.f - ms_graph / ms_serial));
    std::printf("results bit-identical, verification passed\n");

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(capture_stream);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    cudaFree(d_sums);
    std::free(h_x);
    std::free(h_y);
    std::free(h_sums_a);
    std::free(h_sums_b);
    return 0;
}
