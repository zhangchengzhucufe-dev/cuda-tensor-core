// SpMV (sparse matrix-vector multiply, CSR format): sparse computing plus
// warp-level cooperation in one example.
// y[i] = sum_k A[i][k] * x[k], where A stores only nonzeros.
//
// CSR layout: values (the nonzeros), col_idx (their column), row_ptr
// (offset of each row's start in values, M+1 entries).
//
// Two kernels, and which one wins depends on the row length:
//   thread-per-row: one row per thread, serial dot product. Simplest thing
//     that works, best when rows are short (a handful of nonzeros)
//   warp-per-row: one row per warp, 32 lanes stride over the nonzeros,
//     one __shfl_down_sync reduction at the end. With rows of tens of
//     nonzeros the extra parallelism wins big -- so this demo generates
//     ~64 nnz/row
// Picking the parallel granularity to match the workload is the first
// lesson of sparse/graph kernel design.
#include <cstdio>
#include <cstdlib>
#include "error_check.h"

// v1: one row per thread
__global__ void spmv_thread_per_row(const float* values, const int* col_idx,
                                    const int* row_ptr, const float* x,
                                    float* y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.f;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
        sum += values[k] * x[col_idx[k]];
    }
    y[row] = sum;
}

// v2: one row per warp
__global__ void spmv_warp_per_row(const float* values, const int* col_idx,
                                  const int* row_ptr, const float* x,
                                  float* y, int rows) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp >= rows) return;

    int start = row_ptr[warp];
    int end = row_ptr[warp + 1];
    float sum = 0.f;
    // 32 lanes stride over the row's nonzeros (accesses stay coalesced)
    for (int k = start + lane; k < end; k += 32) {
        sum += values[k] * x[col_idx[k]];
    }
    // warp reduction: 5 shuffle rounds, lane 0 ends up with the sum
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[warp] = sum;
}

int main(int argc, char** argv) {
    std::srand(0);
    int rows = 131072;
    int min_nnz = 32, max_nnz = 96;  // avg 64 nnz/row -> ~8.4M nonzeros
    if (argc > 1) rows = std::atoi(argv[1]);
    int cols = rows;  // square-ish

    // ---- build the CSR (row lengths random in [min_nnz, max_nnz)) ----
    long long total_nnz = 0;
    int* h_row_ptr = static_cast<int*>(std::malloc((rows + 1) * sizeof(int)));
    h_row_ptr[0] = 0;
    for (int r = 0; r < rows; ++r) {
        total_nnz += min_nnz + std::rand() % (max_nnz - min_nnz);
        h_row_ptr[r + 1] = static_cast<int>(total_nnz);
    }
    int* h_col_idx = static_cast<int*>(std::malloc(total_nnz * sizeof(int)));
    float* h_values = static_cast<float*>(std::malloc(total_nnz * sizeof(float)));
    for (long long k = 0; k < total_nnz; ++k) {
        h_col_idx[k] = std::rand() % cols;
        h_values[k] = std::rand() / static_cast<float>(RAND_MAX) - 0.5f;
    }
    float* h_x = static_cast<float*>(std::malloc(cols * sizeof(float)));
    float* h_y = static_cast<float*>(std::malloc(rows * sizeof(float)));
    float* h_ref = static_cast<float*>(std::malloc(rows * sizeof(float)));
    fill_random_host(h_x, cols, -1.f, 1.f);

    // CPU reference in double
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int k = h_row_ptr[r]; k < h_row_ptr[r + 1]; ++k)
            sum += static_cast<double>(h_values[k]) * h_x[h_col_idx[k]];
        h_ref[r] = static_cast<float>(sum);
    }

    int *d_col_idx, *d_row_ptr;
    float *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_col_idx, total_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, total_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, total_nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, total_nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));

    CudaTimer timer;
    int block = 256;

    // v1
    timer.start();
    spmv_thread_per_row<<<(rows + block - 1) / block, block>>>(
        d_values, d_col_idx, d_row_ptr, d_x, d_y, rows);
    CHECK_KERNEL_LAUNCH();
    float ms1 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, rows, 1e-4f)) {
        std::printf("thread-per-row FAILED\n");
        return 1;
    }

    // v2: grid sized in warps
    int warps = rows;
    timer.start();
    spmv_warp_per_row<<<(warps * 32 + block - 1) / block, block>>>(
        d_values, d_col_idx, d_row_ptr, d_x, d_y, rows);
    CHECK_KERNEL_LAUNCH();
    float ms2 = timer.stop();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));
    if (!compare_close(h_y, h_ref, rows, 1e-4f)) {
        std::printf("warp-per-row FAILED\n");
        return 1;
    }

    // SpMV traffic: each nonzero reads value + col_idx + one x sample,
    // each row writes once
    double gb = (total_nnz * (4 + 4 + 4) + rows * 4) / 1e9;
    std::printf("matrix: %d rows, %lld nnz (~%.0f per row)\n", rows, total_nnz,
                static_cast<double>(total_nnz) / rows);
    std::printf("thread-per-row: %8.3f ms, %7.1f GB/s\n", ms1, gb / (ms1 / 1e3));
    std::printf("warp-per-row  : %8.3f ms, %7.1f GB/s\n", ms2, gb / (ms2 / 1e3));
    std::printf("both passed (with rows longer than 32, warp granularity wins)\n");

    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    std::free(h_col_idx);
    std::free(h_row_ptr);
    std::free(h_values);
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);
    return 0;
}
