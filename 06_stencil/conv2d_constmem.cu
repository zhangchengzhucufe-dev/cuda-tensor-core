// Constant memory (__constant__): the one memory space the other examples
// haven't touched. It's a 64KB read-only region, cached on the SM, with a
// special broadcast behavior: when every thread in a warp reads the SAME
// address, one fetch serves all 32. When they read different addresses,
// the reads serialize per unique address -- so it's great for "small
// read-only data every thread wants" (conv weights, LUTs) and bad for
// per-thread indexed data.
//
// This runs the same tiled 3x3 convolution from conv2d_tiled.cu twice:
// once with the kernel weights in global memory, once in __constant__,
// and compares. The weights are the textbook use case: 9 floats, every
// thread reads the same 9 values over and over.
//
// Up front: on this card the constant-memory version
// comes out slightly SLOWER (see the printed numbers). On modern
// architectures plain read-only global loads already go through the same
// cache path (that's what __ldg / const __restrict__ hints exploit), and a
// 9-float weight set sits in L1 anyway, so the constant cache buys nothing
// here -- this kernel's time is dominated by the shared-memory tile reads.
// The remaining real advantages of __constant__:
//   - copied once with cudaMemcpyToSymbol, persists across launches
//     (no kernel arg, no re-upload)
//   - guaranteed cached, no dependence on the compiler guessing
// Worth knowing the mechanics even when the payoff is negative.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define TILE 32
#define R 1

// device-side symbol, 9 floats. One copy per module, visible to every
// kernel that declares it
__constant__ float const_kernel[3 * 3];

// weights passed as a plain global pointer (the baseline)
__global__ void conv2d_tiled_global(const float* img, const float* kernel,
                                    float* out, int h, int w) {
    __shared__ float tile[TILE + 2 * R][TILE + 2 * R];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * TILE + tx;
    int tile_elems = (TILE + 2 * R) * (TILE + 2 * R);
    int tile_x0 = blockIdx.x * TILE - R;
    int tile_y0 = blockIdx.y * TILE - R;
    for (int idx = tid; idx < tile_elems; idx += TILE * TILE) {
        int sy = idx / (TILE + 2 * R);
        int sx = idx % (TILE + 2 * R);
        int gy = tile_y0 + sy, gx = tile_x0 + sx;
        tile[sy][sx] = (gy >= 0 && gy < h && gx >= 0 && gx < w) ? img[gy * w + gx] : 0.f;
    }
    __syncthreads();

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;
    if (x >= w || y >= h) return;
    float acc = 0.f;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            acc += tile[ty + dy + R][tx + dx + R] * kernel[(dy + R) * 3 + (dx + R)];
    out[y * w + x] = acc;
}

// identical, except the weights come from the constant symbol. Note it's
// not a kernel parameter -- the symbol is linked in at compile time
__global__ void conv2d_tiled_constmem(const float* img, float* out, int h, int w) {
    __shared__ float tile[TILE + 2 * R][TILE + 2 * R];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * TILE + tx;
    int tile_elems = (TILE + 2 * R) * (TILE + 2 * R);
    int tile_x0 = blockIdx.x * TILE - R;
    int tile_y0 = blockIdx.y * TILE - R;
    for (int idx = tid; idx < tile_elems; idx += TILE * TILE) {
        int sy = idx / (TILE + 2 * R);
        int sx = idx % (TILE + 2 * R);
        int gy = tile_y0 + sy, gx = tile_x0 + sx;
        tile[sy][sx] = (gy >= 0 && gy < h && gx >= 0 && gx < w) ? img[gy * w + gx] : 0.f;
    }
    __syncthreads();

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;
    if (x >= w || y >= h) return;
    float acc = 0.f;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            acc += tile[ty + dy + R][tx + dx + R] * const_kernel[(dy + R) * 3 + (dx + R)];
    out[y * w + x] = acc;
}

void conv2d_cpu(const float* img, const float* kernel, float* out, int h, int w) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float acc = 0.f;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w)
                        acc += img[yy * w + xx] * kernel[(dy + R) * 3 + (dx + R)];
                }
            out[static_cast<size_t>(y) * w + x] = acc;
        }
}

int main(int argc, char** argv) {
    std::srand(0);
    int h = 2048, w = 2048;
    if (argc > 2) {
        h = std::atoi(argv[1]);
        w = std::atoi(argv[2]);
    }
    std::printf("conv 3x3, weights in global vs __constant__: image %d x %d\n", h, w);

    size_t pixels = static_cast<size_t>(h) * w;
    float* h_img = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    float* h_out = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    float* h_ref = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    fill_random_host(h_img, pixels);
    float h_kernel[9] = {0.f, -1.f, 0.f, -1.f, 5.f, -1.f, 0.f, -1.f, 0.f};
    conv2d_cpu(h_img, h_kernel, h_ref, h, w);

    float *d_img, *d_out, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_img, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice));
    // the constant-space copy: goes to the symbol, not to a pointer. This
    // is a one-time setup, later launches don't pass weights at all
    CUDA_CHECK(cudaMemcpyToSymbol(const_kernel, h_kernel, 9 * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    CudaTimer timer;

    // baseline: weights in global memory
    timer.start();
    conv2d_tiled_global<<<grid, block>>>(d_img, d_kernel, d_out, h, w);
    CHECK_KERNEL_LAUNCH();
    float ms_global = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    if (!compare_close(h_out, h_ref, pixels, 1e-4f)) {
        std::printf("global-weights version FAILED\n");
        return 1;
    }

    // weights in constant memory
    timer.start();
    conv2d_tiled_constmem<<<grid, block>>>(d_img, d_out, h, w);
    CHECK_KERNEL_LAUNCH();
    float ms_const = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    if (!compare_close(h_out, h_ref, pixels, 1e-4f)) {
        std::printf("constmem version FAILED\n");
        return 1;
    }

    float gb = 2.f * pixels * sizeof(float) / 1e9f;
    std::printf("global weights : %8.3f ms, %7.1f GB/s\n", ms_global,
                gb / (ms_global / 1e3f));
    std::printf("__constant__   : %8.3f ms, %7.1f GB/s\n", ms_const,
                gb / (ms_const / 1e3f));
    std::printf("both passed (yes, const is the slower one here -- 9 floats sit in\n"
                " L1 either way and this kernel is shared-mem bound; see the file\n"
                " header for when constant memory actually pays)\n");

    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_kernel);
    std::free(h_img);
    std::free(h_out);
    std::free(h_ref);
    return 0;
}
