// Multi-stream overlap: getting copies and kernels to run at the same time.
//
// The default stream serializes everything. Separate (non-default) streams
// can run concurrently, so the classic pattern is to chunk a big array and
// pipeline it: while stream A computes on chunk 1, stream B can already be
// copying chunk 2.
//
// Three things you have to get right for this to work:
//   1. pinned host memory (cudaMallocHost). Pageable memory copies can't go async
//   2. operations inside one stream are ordered; cross-stream dependencies
//      need cudaEvent
//   3. kernel args (pointer offsets) must be computed per chunk
//
// Runs a SAXPY over a big array two ways: one default stream, and 4 streams
// round-robin. Real-world speedup depends on how many copy engines the GPU
// has -- consumer cards usually have one, so overlap gains are modest there.
// Tesla-class cards have several and show the full benefit. A small speedup
// here doesn't mean the code is wrong.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define N_STREAMS 4
#define CHUNK_FRACTION 4  // ~1/4 of the array per chunk (last one short if n % 4)

__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 26;  // 67M elements, 256MB per array
    if (argc > 1) n = std::atoi(argv[1]);
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float alpha = 2.f;
    std::printf("n = %d (%.0f MB per array)\n", n, bytes / 1e6);

    // pinned host memory: cudaMallocHost, not malloc
    float *h_x, *h_y, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    for (int i = 0; i < n; ++i) {
        h_x[i] = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
        h_y[i] = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
    }

    float *d_x, *d_y, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    int block = 256;
    CudaTimer timer;
    // ceil so CHUNK_FRACTION chunks always cover all of n (a plain n/4
    // floor silently dropped the remainder, and the check caught it)
    int chunk = (n + CHUNK_FRACTION - 1) / CHUNK_FRACTION;

    // ---- version A: one default stream, copy -> compute -> copy back, serial ----
    timer.start();
    for (int c = 0; c < CHUNK_FRACTION; ++c) {
        size_t off = static_cast<size_t>(c) * chunk;
        int len = static_cast<int>(static_cast<size_t>(n) - off);
        if (len > chunk) len = chunk;  // last chunk is short when n % 4 != 0
        size_t len_bytes = static_cast<size_t>(len) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_x + off, h_x + off, len_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y + off, h_y + off, len_bytes,
                              cudaMemcpyHostToDevice));
        saxpy<<<(len + block - 1) / block, block>>>(alpha, d_x + off, d_y + off,
                                                    d_out + off, len);
        CHECK_KERNEL_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_out + off, d_out + off, len_bytes,
                              cudaMemcpyDeviceToHost));
    }
    // default stream ops are host-synchronizing anyway, but timing with
    // events keeps it consistent
    float ms_serial = timer.stop();

    // ---- version B: N_STREAMS streams pipelined ----
    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    timer.start();
    for (int c = 0; c < CHUNK_FRACTION; ++c) {
        cudaStream_t stream = streams[c % N_STREAMS];  // round-robin
        size_t off = static_cast<size_t>(c) * chunk;
        int len = static_cast<int>(static_cast<size_t>(n) - off);
        if (len > chunk) len = chunk;
        size_t len_bytes = static_cast<size_t>(len) * sizeof(float);
        // async copy in -> kernel -> async copy back. Same stream, so the
        // ordering within a chunk is guaranteed
        CUDA_CHECK(cudaMemcpyAsync(d_x + off, h_x + off, len_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_y + off, h_y + off, len_bytes,
                                   cudaMemcpyHostToDevice, stream));
        saxpy<<<(len + block - 1) / block, block, 0, stream>>>(
            alpha, d_x + off, d_y + off, d_out + off, len);
        CHECK_KERNEL_LAUNCH();
        CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, len_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    // wait for all streams (event-based, more precise than a full device sync)
    for (int s = 0; s < N_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
    float ms_pipelined = timer.stop();

    float max_err = 0.f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(h_out[i] - (alpha * h_x[i] + h_y[i])));
    }
    std::printf("single stream: %8.3f ms\n", ms_serial);
    std::printf("multi stream : %8.3f ms  (speedup %.2fx)\n", ms_pipelined,
                ms_serial / ms_pipelined);
    std::printf("max err: %g, %s\n", max_err,
                max_err < 1e-5f ? "verification passed" : "verification FAILED");
    if (max_err >= 1e-5f) return 1;

    for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_out);
    return 0;
}
