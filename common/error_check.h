// Shared helpers: error checking, event timing, random data, result comparison.
// Every example here includes this so I don't have to repeat the boilerplate.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every cuda* API call with this. Bails out with file/line on failure.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(err__),       \
                         static_cast<int>(err__));                            \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Catches a bad launch itself (bad grid dims, too many args, etc).
// Runtime errors *inside* the kernel only show up later, at the next sync.
#define CHECK_KERNEL_LAUNCH()                                                 \
    do {                                                                      \
        cudaError_t err__ = cudaGetLastError();                               \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "Kernel launch failed at %s:%d: %s\n",       \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Times a chunk of GPU work with two cuda events, returns milliseconds.
class CudaTimer {
   public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

   private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Random floats in [lo, hi), filled on the host.
inline void fill_random_host(float* h, size_t n, float lo = 0.f, float hi = 1.f) {
    for (size_t i = 0; i < n; ++i) {
        h[i] = lo + (hi - lo) * (std::rand() / static_cast<float>(RAND_MAX));
    }
}

// Element-wise float compare: |a-b| <= tol * max(1, |b|).
// Prints the first mismatch and returns false, which is all I ever need.
inline bool compare_close(const float* a, const float* b, size_t n, float tol) {
    for (size_t i = 0; i < n; ++i) {
        float limit = tol * fmaxf(1.f, fabsf(b[i]));
        if (fabsf(a[i] - b[i]) > limit) {
            std::fprintf(stderr,
                         "mismatch at i=%zu: got %f, expect %f (limit %f)\n", i,
                         a[i], b[i], limit);
            return false;
        }
    }
    return true;
}
