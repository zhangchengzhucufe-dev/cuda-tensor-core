// The lab kernels as torch ops -- the actual shape of kernel-engineering
// work: same kernels as the standalone examples, but wrapped the way a
// real inference engine consumes them (ATen tensors in, ATen tensor out,
// TORCH_CHECK contracts at the boundary, torch manages the memory).
//
//   mm(a, b)        -- the 8x8 register-tile SGEMM from 08_gemm_opt
//   layernorm(x,w,b)-- the two-block-reduction layernorm from 09_nn_ops
//
// Non-negotiable constraints, enforced loudly: mm needs M%128==0, N%128==0,
// K%8==0 (that's the tile shape, and padding inside the wrapper would hide
// the constraint that makes the kernel fast). build_and_benchmark.py
// checks both ops against torch's own and times the gap.
//
// Built with torch's cpp_extension, not the repo Makefile:
//   python3 build_and_benchmark.py
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_regtile(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float acc[TM][TN] = {};
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    for (int t = 0; t < K / BK; ++t) {
        int a_row = ty * 8 + tx / 2;
        int a_col = (tx % 2) * 4;
        float4 av = *reinterpret_cast<const float4*>(
            A + (row_base + a_row) * K + t * BK + a_col);
        As[a_row][a_col] = av.x;
        As[a_row][a_col + 1] = av.y;
        As[a_row][a_col + 2] = av.z;
        As[a_row][a_col + 3] = av.w;

        int tid = ty * 16 + tx;
        int b_row = tid / 32;
        int b_col = (tid % 32) * 4;
        float4 bv = *reinterpret_cast<const float4*>(
            B + (t * BK + b_row) * N + col_base + b_col);
        Bs[b_row][b_col] = bv.x;
        Bs[b_row][b_col + 1] = bv.y;
        Bs[b_row][b_col + 2] = bv.z;
        Bs[b_row][b_col + 3] = bv.w;
        __syncthreads();

        float a_frag[TM], b_frag[TN];
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[i] = As[ty * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(row_base + ty * TM + i) * N + col_base + tx * TN + j] = acc[i][j];
}

// block-per-row layernorm: two block reductions (sum, sum of squares),
// variance via E[x^2] - mean^2 -- the choice 09_nn_ops/layernorm.cu
// documents and defends.
__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y, int D, float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * D;
    float* yr = y + static_cast<size_t>(blockIdx.x) * D;
    int tid = threadIdx.x;

    float sum = 0.f, sumsq = 0.f;
    for (int d = tid; d < D; d += blockDim.x) {
        float v = xr[d];
        sum += v;
        sumsq += v * v;
    }
    // block reduce (sum, sumsq) together: warp shuffles then one smem round
    __shared__ float wsum[32], wsumsq[32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, off);
    }
    if (tid % 32 == 0) {
        wsum[tid / 32] = sum;
        wsumsq[tid / 32] = sumsq;
    }
    __syncthreads();
    if (tid == 0) {
        float ts = 0.f, tsq = 0.f;
        for (int w = 0; w < blockDim.x / 32; ++w) {
            ts += wsum[w];
            tsq += wsumsq[w];
        }
        float mean = ts / D;
        wsum[0] = mean;
        wsum[1] = rsqrtf(tsq / D - mean * mean + eps);
    }
    __syncthreads();
    float mean = wsum[0], rstd = wsum[1];
    for (int d = tid; d < D; d += blockDim.x)
        yr[d] = (xr[d] - mean) * rstd * weight[d] + bias[d];
}

static void check_cuda_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
}

torch::Tensor mm(torch::Tensor a, torch::Tensor b) {
    check_cuda_2d(a, "a");
    check_cuda_2d(b, "b");
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    int64_t M = ac.size(0), K = ac.size(1), N = bc.size(1);
    TORCH_CHECK(bc.size(0) == K, "shape mismatch: ", M, "x", K, " @ ", K, "x",
                N, " wanted");
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0,
                "regtile needs M%128==0, N%128==0, K%8==0; got ", M, "x", N,
                "x", K, " (padding is deliberately your problem)");
    auto c = torch::empty({M, N}, ac.options());
    dim3 block(16, 16);
    dim3 grid(N / BN, M / BM);
    sgemm_regtile<<<grid, block>>>(ac.data_ptr<float>(), bc.data_ptr<float>(),
                                   c.data_ptr<float>(), static_cast<int>(M),
                                   static_cast<int>(N), static_cast<int>(K));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mm launch failed: ", cudaGetErrorString(err));
    return c;
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor weight,
                        torch::Tensor bias) {
    check_cuda_2d(x, "x");
    TORCH_CHECK(weight.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat,
                "weight/bias must be float32");
    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto bc = bias.contiguous();
    int64_t S = xc.size(0), D = xc.size(1);
    TORCH_CHECK(wc.numel() == D && bc.numel() == D, "weight/bias need ", D,
                " elements");
    TORCH_CHECK(D <= 1024 * 32, "D too big for one-block-per-row");
    auto y = torch::empty_like(xc);
    int threads = 256;  // D is strided inside, so block size stays fixed
    layernorm_kernel<<<static_cast<unsigned>(S), threads>>>(
        xc.data_ptr<float>(), wc.data_ptr<float>(), bc.data_ptr<float>(),
        y.data_ptr<float>(), static_cast<int>(D), 1e-5f);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "layernorm launch failed: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm", &mm, "register-tile sgemm (M%128==0, N%128==0, K%8==0)");
    m.def("layernorm", &layernorm, "fused layernorm, block per row");
}
