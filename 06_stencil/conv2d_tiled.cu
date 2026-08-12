// 2D convolution (stencil computation): the example for halo loading.
// 3x3 kernel over an image, zero-padded borders.
//
// Every 3x3 neighborhood is 9 reads, and neighboring threads' windows
// overlap heavily -- the naive version pulls the image from global memory
// 9x. With shared memory tiling:
//   - the block's 32x32 threads cooperatively load a (32+2)x(32+2) region
//   - interior threads load their own pixel, the border ring handles the halo
//   - afterwards all 9 neighborhood reads hit shared memory
//
// The halo load uses the "linear sweep" pattern: spread the (TILE+2R)^2
// elements over TILE^2 threads, each thread figures out its slot with a
// divide/mod. No special-casing the four edges and four corners -- works
// for any halo radius.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

#define TILE 32
#define R 1  // kernel radius; a 3x3 kernel means R = 1

__global__ void conv2d_naive(const float* img, const float* kernel,
                             float* out, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float acc = 0.f;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            // zero padding: out-of-bounds reads contribute nothing
            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                acc += img[yy * w + xx] * kernel[(dy + R) * (2 * R + 1) + (dx + R)];
            }
        }
    }
    out[y * w + x] = acc;
}

__global__ void conv2d_tiled(const float* img, const float* kernel,
                             float* out, int h, int w) {
    __shared__ float tile[TILE + 2 * R][TILE + 2 * R];

    // --- halo load: spread (TILE+2R)^2 elements over TILE^2 threads ---
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * TILE + tx;
    int tile_elems = (TILE + 2 * R) * (TILE + 2 * R);
    int tile_x0 = blockIdx.x * TILE - R;  // top-left of the shared tile in input coords
    int tile_y0 = blockIdx.y * TILE - R;
    for (int idx = tid; idx < tile_elems; idx += TILE * TILE) {
        int sy = idx / (TILE + 2 * R);  // row/col within the shared tile
        int sx = idx % (TILE + 2 * R);
        int gy = tile_y0 + sy;          // matching global coords
        int gx = tile_x0 + sx;
        if (gy >= 0 && gy < h && gx >= 0 && gx < w) {
            tile[sy][sx] = img[gy * w + gx];
        } else {
            tile[sy][sx] = 0.f;  // zero padding
        }
    }
    __syncthreads();

    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;
    if (x >= w || y >= h) return;

    float acc = 0.f;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            acc += tile[ty + dy + R][tx + dx + R] * kernel[(dy + R) * (2 * R + 1) + (dx + R)];
        }
    }
    out[y * w + x] = acc;
}

void conv2d_cpu(const float* img, const float* kernel, float* out, int h, int w) {
    for (int y = 0; y < h; ++y) {
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
}

int main(int argc, char** argv) {
    std::srand(0);
    int h = 2048, w = 2048;
    if (argc > 2) {
        h = std::atoi(argv[1]);
        w = std::atoi(argv[2]);
    }
    std::printf("image %d x %d (%.1f MB), 3x3 kernel\n", h, w, h * w * 4. / 1e6);

    size_t pixels = static_cast<size_t>(h) * w;
    float* h_img = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    float* h_out = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    float* h_ref = static_cast<float*>(std::malloc(pixels * sizeof(float)));
    fill_random_host(h_img, pixels);

    // some arbitrary sharpen-ish kernel
    float h_kernel[9] = {0.f, -1.f, 0.f, -1.f, 5.f, -1.f, 0.f, -1.f, 0.f};

    float *d_img, *d_out, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_img, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img, pixels * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    CudaTimer timer;

    timer.start();
    conv2d_naive<<<grid, block>>>(d_img, d_kernel, d_out, h, w);
    CHECK_KERNEL_LAUNCH();
    float ms_naive = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // one CPU pass serves as reference for both GPU versions
    conv2d_cpu(h_img, h_kernel, h_ref, h, w);
    for (size_t i = 0; i < pixels; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f * (1.f + std::fabs(h_ref[i]))) {
            std::fprintf(stderr, "naive FAILED at %zu\n", i);
            return 1;
        }
    }

    timer.start();
    conv2d_tiled<<<grid, block>>>(d_img, d_kernel, d_out, h, w);
    CHECK_KERNEL_LAUNCH();
    float ms_tiled = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < pixels; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f * (1.f + std::fabs(h_ref[i]))) {
            std::fprintf(stderr, "tiled FAILED at %zu: got %f expect %f\n", i,
                         h_out[i], h_ref[i]);
            return 1;
        }
    }

    float gb = 2.f * pixels * sizeof(float) / 1e9f;  // read once + write once
    std::printf("naive: %8.3f ms, %7.1f GB/s\n", ms_naive, gb / (ms_naive / 1e3f));
    std::printf("tiled: %8.3f ms, %7.1f GB/s  (halo loaded into shared memory)\n",
                ms_tiled, gb / (ms_tiled / 1e3f));
    std::printf("verification passed\n");

    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_kernel);
    std::free(h_img);
    std::free(h_out);
    std::free(h_ref);
    return 0;
}
