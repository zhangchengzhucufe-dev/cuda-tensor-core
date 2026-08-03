// Unified memory (cudaMallocManaged):
//   - one pointer, dereferenceable on both host and device, pages migrate
//     on demand
//   - the upside: no cudaMemcpy to write. The downside: page faults on
//     first touch, and you don't control when migrations happen unless you
//     prefetch explicitly
//
// This demo does three things:
//   1. runs a SAXPY through managed memory and checks it
//   2. prefetches the buffers to the GPU with cudaMemPrefetchAsync so the
//      page faults don't land inside the kernel
//   3. compares kernel time with and without the prefetch
//
// Heads up: WSL2 doesn't do page migration at all, the prefetch call just
// errors out there. The code detects that and moves on. On native Linux
// you'll see the real difference.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

__global__ void saxpy(float a, const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

// CUDA 13 changed cudaMemPrefetchAsync to take a cudaMemLocation struct.
// device >= 0 means migrate to that GPU, device < 0 means back to the CPU.
// Returns false when prefetch isn't supported (WSL2, basically).
bool prefetch(const void* p, size_t bytes, int device) {
    cudaMemLocation loc{};
    if (device >= 0) {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    } else {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    }
    cudaError_t err = cudaMemPrefetchAsync(p, bytes, loc, 0);
    if (err != cudaSuccess) {
        // clear the sticky last-error state, otherwise it pollutes the next
        // CHECK_KERNEL_LAUNCH
        cudaGetLastError();
        std::printf("(prefetch not supported here: %s, skipping -- common on WSL2)\n",
                    cudaGetErrorString(err));
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 24;
    if (argc > 1) n = std::atoi(argv[1]);
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float alpha = 2.0f;
    std::printf("n = %d, %.1f MB of managed memory total\n", n, 3.0 * bytes / 1e6);

    // one cudaMallocManaged replaces cudaMalloc + cudaMemcpy
    float *x, *y, *out;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    CUDA_CHECK(cudaMallocManaged(&out, bytes));
    for (int i = 0; i < n; ++i) {
        x[i] = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
        y[i] = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
    }

    int block = 256;
    int grid = (n + block - 1) / block;
    CudaTimer timer;

    // Version A: data was just written by the CPU, so the kernel's accesses
    // trigger page faults all over the place
    timer.start();
    saxpy<<<grid, block>>>(alpha, x, y, out, n);
    CHECK_KERNEL_LAUNCH();
    float ms_cold = timer.stop();

    float max_err = 0.f;
    for (int i = 0; i < n; ++i) {
        float expect = alpha * x[i] + y[i];
        max_err = fmaxf(max_err, fabsf(out[i] - expect));
    }
    std::printf("version A (no prefetch): %.3f ms, max err %g\n", ms_cold, max_err);
    if (max_err > 1e-5f) {
        std::printf("FAILED\n");
        return 1;
    }

    // Version B: prefetch everything to the GPU first, then run
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    bool did_prefetch = true;
    did_prefetch &= prefetch(x, bytes, device);
    did_prefetch &= prefetch(y, bytes, device);
    did_prefetch &= prefetch(out, bytes, device);

    timer.start();
    saxpy<<<grid, block>>>(alpha, x, y, out, n);
    CHECK_KERNEL_LAUNCH();
    float ms_warm = timer.stop();
    // only a fair comparison if the prefetch actually happened; on WSL2 the
    // second run is faster too, just because everything is resident/warm
    std::printf("version B (prefetched) : %.3f ms%s\n", ms_warm,
                did_prefetch ? "" : " (prefetch didn't happen here, take with a grain of salt)");

    prefetch(out, bytes, -1);  // migrate back if the host is going to read it

    cudaFree(x);
    cudaFree(y);
    cudaFree(out);
    return 0;
}
