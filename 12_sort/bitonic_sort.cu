// Bitonic sort: the classic teaching sort for GPUs.
// A bitonic sequence (rises then falls) can be merged into sorted order by
// log(n) steps of compare-exchange; the whole sort is log(n) rounds of
// that = O(n log^2 n) comparisons. Every step is branch-free data-parallel
// compare-and-swap though, which is exactly what GPUs like.
//
// This is the standard "global iteration" version: the host runs the double
// loop and launches a kernel per step, each thread handles one pair.
//   - i pairs with i^j (xor guarantees unique, complete pairing)
//   - blocks where (i & k) == 0 sort ascending, the rest descending
//
// Every swap goes through global memory and there are log^2 kernel
// launches, so this won't beat std::sort -- the value is seeing the shape
// of the sorting network. Production code sorts 1K-4K chunks in shared
// memory inside one kernel first, or just uses cub::DeviceRadixSort.
//
// Bitonic needs a power-of-two length. If you pass something else this
// demo just rounds n up to the next power of two and sorts that many
// elements (production code would pad with +inf and keep the real length).
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "error_check.h"

// One compare-exchange step: j is the pairing distance, k is the current
// block length (which sets the sort direction)
__global__ void bitonic_step(float* data, int j, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int partner = i ^ j;        // this thread's partner
    if (partner <= i) return;   // only the lower index of the pair acts, no double writes

    bool ascending = (i & k) == 0;
    float vi = data[i];
    float vp = data[partner];
    // swap in an ascending block when vi > vp, in a descending one when
    // vi < vp -- which happens to be the same condition
    if ((vi > vp) == ascending) {
        data[i] = vp;
        data[partner] = vi;
    }
}

int main(int argc, char** argv) {
    std::srand(0);
    int n = 1 << 20;  // 1M elements, must be a power of two
    if (argc > 1) {
        n = std::atoi(argv[1]);
        int p = 1;
        while (p < n) p <<= 1;
        n = p;
    }
    std::printf("bitonic sort: n = %d\n", n);

    float* h_data = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_ref = static_cast<float*>(std::malloc(n * sizeof(float)));
    fill_random_host(h_data, n, -1000.f, 1000.f);
    std::copy(h_data, h_data + n, h_ref);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;
    CudaTimer timer;
    timer.start();
    // outer k: bitonic block length doubles from 2 to n
    // inner j: merge distance halves from k/2 down to 1
    int launches = 0;
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_step<<<grid, block>>>(d_data, j, k, n);
            ++launches;
        }
    }
    CHECK_KERNEL_LAUNCH();
    float ms = timer.stop();
    std::printf("kernel: %.3f ms, %d kernel launches total\n", ms, launches);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::sort(h_ref, h_ref + n);
    for (int i = 0; i < n; ++i) {
        if (h_data[i] != h_ref[i]) {
            std::fprintf(stderr, "FAILED at %d: got %f, expect %f\n", i,
                         h_data[i], h_ref[i]);
            return 1;
        }
    }
    std::printf("verification passed (matches std::sort exactly)\n");

    cudaFree(d_data);
    std::free(h_data);
    std::free(h_ref);
    return 0;
}
