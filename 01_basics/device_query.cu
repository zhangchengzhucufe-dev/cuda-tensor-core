// Device query. Always worth running before you write any kernel --
// the launch config you can pick, and how fast you can possibly go,
// are both decided by these numbers.
#include <cstdio>
#include "error_check.h"

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::printf("No CUDA device found\n");
        return 1;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::printf("=== device %d: %s ===\n", dev, prop.name);
        std::printf("  compute capability       : %d.%d\n", prop.major, prop.minor);
        std::printf("  SMs                      : %d\n", prop.multiProcessorCount);
        std::printf("  max threads per SM       : %d\n", prop.maxThreadsPerMultiProcessor);
        std::printf("  max threads per block    : %d\n", prop.maxThreadsPerBlock);
        std::printf("  max block dims           : (%d, %d, %d)\n", prop.maxThreadsDim[0],
                    prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        std::printf("  max grid dims            : (%d, %d, %d)\n", prop.maxGridSize[0],
                    prop.maxGridSize[1], prop.maxGridSize[2]);
        std::printf("  warp size                : %d\n", prop.warpSize);
        std::printf("  global memory            : %.1f GB\n",
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  shared mem per block     : %zu KB\n", prop.sharedMemPerBlock / 1024);
        std::printf("  shared mem per SM        : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        std::printf("  L2 cache                 : %d MB\n", prop.l2CacheSize / (1024 * 1024));
        std::printf("  constant memory          : %zu KB\n", prop.totalConstMem / 1024);
        // CUDA 13 dropped a bunch of fields from cudaDeviceProp, so these come
        // from the attribute API instead
        int clock_khz = 0, overlap = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, dev));
        std::printf("  clock                    : %.2f GHz\n", clock_khz / 1e6);
        std::printf("  overlapping copy/compute : %s\n", overlap ? "yes" : "no");
        std::printf("  unified addressing       : %s\n",
                    prop.unifiedAddressing ? "yes" : "no");

        // Rough FP32 peak from SM count x clocks. Ignores tensor cores,
        // it's just a sanity number to compare my kernel timings against.
        float cores_per_sm = 0.f;  // different per architecture
        if (prop.major == 8) cores_per_sm = 128.f;      // Ampere
        else if (prop.major == 9) cores_per_sm = 128.f; // Hopper (Ada is 8.x, caught above)
        else if (prop.major == 7) cores_per_sm = 64.f;  // Volta/Turing
        if (cores_per_sm > 0.f) {
            float tflops = prop.multiProcessorCount * cores_per_sm * 2.f *
                           clock_khz / 1e9;
            std::printf("  rough FP32 peak          : %.2f TFLOPS (no tensor cores)\n",
                        tflops);
        }
        std::printf("\n");
    }
    return 0;
}
