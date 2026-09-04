# Profiling notes

Everything below is from the actual card this repo was developed on
(RTX 3060 Laptop, WSL2, CUDA 13.3). Raw command + raw output, then what
I read out of it.

## What works where (WSL2)

- `nsys` works: kernel timelines, API call costs, memcpy traffic. Used below.
- `ncu` does **not** work here: hardware performance counters are blocked
  by the driver, every profile dies with

  ```
  ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
  NVIDIA GPU Performance Counters on the target device 0.
  ```

  On native Linux the fix is the `NVreg_RestrictProfilingToAdminUsers=0`
  module option; in WSL2 there's no equivalent, counters are simply off.
  So all counter-level analysis (occupancy, bank conflicts, roofline) has
  to happen on a real Linux box or the DCA cluster. Commands I'd run there
  are at the bottom.

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

## What I'd run on a counter-enabled box

```
# occupancy + register pressure on the GEMM ladder kernels
ncu --set full --kernel-name regex:sgemm ./build/08_gemm_opt/sgemm_register_tile

# the two questions worth answering for the register-tile kernel:
#   1. launch__registers_per_thread -> why 6 blocks/SM and not more
#   2. sm__pipe_tensor / sm__throughput -> how close to issue-bound

# bank conflicts on the transpose / reduction kernels
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./build/02_memory/transpose_tiled

# roofline position of the bandwidth-bound kernels
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/01_basics/vector_add
```

Expected (from the literature, to be confirmed): register-tile GEMM sits
at ~128 regs/thread -> 6 blocks/SM on sm_86; vector_add should show
memory throughput near peak and SM throughput in single digits, the
signature of a bandwidth-bound kernel.
