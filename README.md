# CUDA Tensor Core Lab

[![build](https://github.com/zhangchengzhucufe-dev/cuda-tensor-core-lab/actions/workflows/build.yml/badge.svg)](https://github.com/zhangchengzhucufe-dev/cuda-tensor-core-lab/actions/workflows/build.yml)

A bunch of CUDA kernels I wrote while working through the standard material
(PMPP 3rd ed., the GPU-mode lectures, assorted GEMM blogs). Everything runs
and self-checks against a CPU reference, and every example prints real
timings so you can see the optimization actually pay off. Difficulty is
intermediate: shared memory, warp intrinsics, atomics, WMMA tensor cores,
cp.async, graphs -- no hand-written PTX, no CUTLASS.

Hardware these were tested on: RTX 3060 Laptop (Ampere, `sm_86`), CUDA 13.x.
Numbers below are from this card; yours will differ.

## Build & run

```bash
make          # builds everything into build/
make run      # builds + runs all examples, each one verifies itself
make clean
```

Individual examples take size args if you want them:

```bash
./build/02_memory/transpose_tiled 8192
./build/03_gemm/sgemm_tiled 2048 2048 2048
```

## What's in here

| dir | file | the point |
|---|---|---|
| `common/` | `error_check.h` | `CUDA_CHECK` macro, event timer, result comparison -- shared by everything |
| `01_basics/` | `device_query.cu` | SM count, shared memory limits, etc. The numbers every launch config decision comes from |
| | `vector_add.cu` | smallest possible kernel, plus the bandwidth math: 1 FLOP per 12 bytes = memory bound |
| `02_memory/` | `transpose_tiled.cu` | coalescing: a transpose always loses on one side until you use shared memory. `[TILE][TILE+1]` kills bank conflicts |
| | `reduction.cu` | three versions of sum: divergent -> interleaved -> warp shuffles + coarsening. Same FLOPs, ~6x apart |
| | `unified_memory.cu` | `cudaMallocManaged`, page migration cost, `cudaMemPrefetchAsync` |
| `03_gemm/` | `sgemm_naive.cu` | zero data reuse, single-digit % of peak |
| | `sgemm_tiled.cu` | shared memory tiling, the foundation cuBLAS/CUTLASS build on |
| | `sgemm_wmma.cu` | tensor cores via WMMA: fragments, `mma_sync`, FP16 in / FP32 accumulate |
| `04_scan/` | `prefix_sum.cu` | work-efficient Blelloch scan + the three-pass trick for arrays bigger than a block |
| `05_histogram/` | `histogram.cu` | atomics: global `atomicAdd` contention -> shared memory privatization + coarsening (~17x) |
| `06_stencil/` | `conv2d_tiled.cu` | 2D conv with halo loading; the divide/mod pattern handles edges and corners with no special cases |
| | `conv2d_constmem.cu` | `__constant__` weights: cudaMemcpyToSymbol, warp broadcast semantics -- and a measured case where it doesn't help |
| `07_streams/` | `stream_overlap.cu` | pinned memory + `cudaMemcpyAsync` + multi-stream pipelining |
| `08_gemm_opt/` | `sgemm_register_tile.cu` | 2D register tiling (8x8 per thread) + `float4` loads: the biggest single step in the GEMM ladder |
| | `sgemm_double_buffer.cu` | cp.async double buffering -- and why it *didn't* help at BK=8 (honest negative result, notes in the file) |
| | `sgemm_vs_cublas.cu` | same data, me vs cuBLAS. Reaches ~71% of cuBLAS at 2048^3 on this card |
| `09_nn_ops/` | `softmax_rowwise.cu` | safe softmax, thread-per-row vs block-per-row; reusable block reduction helper |
| | `layernorm.cu` | two block reductions, `E[x^2] - mean^2` variance, and why that's okay here |
| | `bias_gelu_fused.cu` | kernel fusion 101: skip a full read+write round trip of the intermediate |
| `10_attention/` | `attention_forward.cu` | online softmax (the Flash Attention trick): m/l/acc streaming update, no S x S score matrix in memory |
| `11_sparse/` | `spmv_csr.cu` | CSR SpMV, thread-per-row vs warp-per-row -- pick granularity to match row length |
| `12_sort/` | `bitonic_sort.cu` | branch-free compare-exchange network, O(n log^2 n), why GPUs tolerate that |
| `13_graphs/` | `cuda_graphs.cu` | stream capture + graph replay; kills most of the launch overhead for tiny kernels |
| `14_transformer_block/` | `transformer_block.cu` | a full pre-norm block forward (ln -> qkv -> 4-head online-softmax attention -> proj -> residual -> mlp), all kernels from the earlier dirs as stages, verified end-to-end against a double-precision CPU pass, plus eager-vs-cuda-graph replay of the whole sequence |
| `docs/` | `profiling.md` | nsys runs on the real card: what the timings are made of, and the ncu commands I'd run on a counter-enabled box |

## Numbers worth knowing (RTX 3060 Laptop)

- reduction: divergent 35.7 GB/s -> warp shuffles 227.7 GB/s
- transpose: naive 73 GB/s -> tiled 190 GB/s
- GEMM at 2048^3: naive 536 GFLOPS -> register-tile ~4.7 TFLOPS -> cuBLAS ~6.6 TFLOPS
- histogram: naive 6.8 ms -> privatized 0.4 ms
- bias+GELU fusion: 2.0 ms -> 0.64 ms (memory bound, so fusion ≈ 2x)
- CUDA graphs: ~40-70% total time saved on a 3-tiny-kernel pipeline; on
  the transformer block (kernels with real work) only ~1.07x -- knowing
  when a tool doesn't apply is half the point

## Two quirks of this test machine (native Linux won't have these)

- WSL2 doesn't support unified memory page migration: `cudaMemPrefetchAsync`
  returns `invalid device ordinal`. `unified_memory.cu` detects it and moves on.
- Consumer GPUs have a single copy engine, so the multi-stream example only
  gets ~1.03x. That's a hardware limit, not a bug.

## Bugs I actually hit while writing these

- block reductions: a `__shfl_sync` broadcast only crosses lanes, not warps.
  Warps 1-7 silently read garbage. Broadcast through shared memory instead.
- scan: the second pass needs an *exclusive* scan of block sums. I wrote an
  inclusive one and block 0 was off by exactly its own sum.
- non-deterministic float accumulation (`sums[bid] += out[i]` from 256
  racing threads) made my graphs example un-verifiable. Tree-reduce in
  shared memory instead.
- cp.async faults with "misaligned address" if the shared tile pad isn't a
  multiple of 4 floats: the `[BK][BN+1]` trick from the transpose example
  breaks 16B alignment. Pad to +4 there.
- transformer block: the bias-add after the qkv GEMM launched with a grid
  sized for S*D elements over an S*3*D buffer -- only the Q slice got its
  bias. Underlaunches like this don't crash, they just produce quietly
  wrong numbers. Found it by diffing stages against the CPU reference.
- same file, the CPU reference itself used the layernorm output as Q
  instead of the projected Q. The GPU was right and the reference was
  wrong, which is its own lesson: verify the verifier.

## Ideas if you want to go further

- 3-4 stage cp.async pipeline or BK=16, to make the double-buffer version
  actually pay (see the notes in the file for why 2 stages at BK=8 doesn't)
- multi-query-per-block attention, then flash style tiling over keys
- wrap the register-tile GEMM and layernorm as a torch extension and
  benchmark against `torch.nn`
- counter-level analysis needs a non-WSL2 machine: the exact ncu commands
  are waiting in docs/profiling.md
