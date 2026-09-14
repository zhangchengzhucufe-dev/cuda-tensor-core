# Profiling notes

Everything below is from the actual card this repo was developed on
(RTX 3060 Laptop, WSL2, CUDA 13.3). Raw command + raw output, then what
I read out of it.

## What works where (WSL2)

- `nsys` works: kernel timelines, API call costs, memcpy traffic. Used below.
- `ncu`: when these notes were first written, hardware performance counters
  were blocked on this machine (`ERR_NVGPUCTRPERM` on every profile). Some
  time between then and 2026-09-13 that stopped being true -- driver update
  on the Windows side or the counter-access setting flipping -- and `ncu
  --set full` now works, permissions and all. The counter-level analysis
  that used to live in "commands I'd run someday" is now real; the full
  GEMM ladder is done in [gemm_deep_dive.md](gemm_deep_dive.md).

  Lesson I'm keeping either way: "this profiler doesn't work here" is a
  dated observation, not a fact about the machine. Re-check the premise
  when you come back to it.

## GEMM: where the time actually goes

```
$ nsys profile -o gemm_reg ./build/08_gemm_opt/sgemm_register_tile
$ nsys stats --report cuda_gpu_kern_sum gemm_reg.nsys-rep

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)
    100.0          7313293          2  3656646.5  3656646.5   3653335   3659958
    Name: sgemm_regtile(...)
```

Two runs, 3.66 ms average, 4.6-4.7 TFLOPS at 2048^3. The min/max spread is
~2 us (0.06%), so the event-timing numbers in the README are trustworthy,
not clock-noise artifacts.

```
$ nsys stats --report cuda_api_sum gemm_reg.nsys-rep

 Time (%)  Total      Num   Avg        Name
     88.1  188.9 ms     3   63.0 ms    cudaMalloc
      6.8   14.6 ms     3    4.9 ms    cudaMemcpy
      0.7    1.5 ms     2  746.1 us    cudaLaunchKernel
```

For one GEMM run, setup (malloc + copies) costs ~10x the kernel itself.
Fine for a benchmark, a disaster for a real serving loop -- which is why
real engines allocate everything up front and never touch malloc in the
hot path. Good thing to have confirmed rather than assumed.

## CUDA graphs: what the 40-70% actually is

```
$ nsys stats --report cuda_api_sum graphs_prof.nsys-rep

 Time (%)  Total      Num   Avg       Name
      2.8  5.81 ms    603    9.6 us   cudaLaunchKernel
      0.9  1.77 ms    200    8.9 us   cudaGraphLaunch
```

Per-call cost is almost the same (9.6 vs 8.9 us). The win is that one
`cudaGraphLaunch` submits 3 kernels while one `cudaLaunchKernel` submits
one -- so the CPU-side submit cost per frame drops from ~29 us to ~9 us.
The GPU kernels didn't get faster; the CPU stopped being the bottleneck.
`nsys` makes the mechanism obvious in a way the wall-clock number alone
doesn't.

## Multi-stream: the copy engine ceiling

```
$ nsys stats --report cuda_gpu_mem_time_sum streams_prof.nsys-rep

 Time (%)  Total (ns)  Count  Avg (ns)   Operation
     67.7  92.19 ms       16   5.76 ms   [CUDA memcpy Host-to-Device]
```

16 H2D copies (4 chunks x 2 arrays... plus D2H on the same engine). The
transfers serialize on a single copy engine, which is the hardware reason
the multi-stream example only buys ~1.03x on this card. On an A100/H100
with multiple engines the same code overlaps properly.

## The counter-level follow-ups, now that they run

**GEMM ladder, full `--set full` treatment** (naive / tiled /
register-tile / double-buffer / CUTLASS / cuBLAS at a uniform 2048^3):
see [gemm_deep_dive.md](gemm_deep_dive.md). The one-line summary: naive
is LSU-issue-bound (`lg_throttle` 23.4 cycles/issue, DRAM only 52%),
register tiling moves the bottleneck to the FMA pipe at a self-inflicted
33% occupancy (128 regs/thread -> 2 blocks/SM), cp.async buys +15% on
top, CUTLASS's stock SIMT config hits 67% of FP32 peak and beats
cuBLAS by 11% on this card.

**Transpose bank conflicts** -- the `[TILE][TILE+1]` pad, verified:

```
$ ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./build/02_memory/transpose_tiled
  op_ld.sum   95141     (tiled kernel)
  op_st.sum   62295     (tiled kernel)
```

95k + 62k conflicts looks scary until you divide: n=4096 means 16384
blocks x 32 warps x 64 shared accesses = 33.5M wavefronts, so ~0.47%
conflict rate -- the pad killed them, and the residue is the edge
blocks where tiles run off the matrix. The naive kernel in the same
binary shows 0/0 because it never touches shared memory at all.

**vector_add, the bandwidth-bound signature**:

```
$ ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/01_basics/vector_add
  dram__throughput...            %   93.96
  gpu__compute_memory_throughput %   93.96
  sm__throughput...              %   15.21
```

DRAM pinned at 94% of peak while the SMs loaf at 15% -- the exact shape
a bandwidth-bound kernel is supposed to have. The README's bandwidth
math now has the counters to back it.
