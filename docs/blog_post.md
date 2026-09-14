# Optimizing a CUDA GEMM from 605 GFLOPS to 20 TFLOPS: what the counters actually said

*Alternate titles I considered: "My 'memory-bound' GEMM wasn't memory-bound" and "A GEMM optimization ladder, profiled on an RTX 3060". Pick whichever fits the platform.*

---

Over the past few weeks I took a hand-written SGEMM from a naive kernel (605 GFLOPS) to FP16 tensor cores (19.6 TFLOPS) on one RTX 3060 Laptop, at M=N=K=2048. That final number is 3x cuBLAS SGEMM on the same card.

The number isn't the point -- anyone with an H100 can do better. What I think is worth sharing is how many of my assumptions the profiler overturned along the way. Three examples up front:

1. My naive kernel was **not** memory-bound. DRAM sat at 52% utilization; what saturated was the load-issue path.
2. My register-tiled kernel runs at 33% occupancy, and that's not a bug -- it's a deliberate trade I didn't understand until the counters forced me to.
3. cuBLAS was not the fastest thing on the card. CUTLASS's stock SIMT config beat it by 11% at this size.

Everything below was measured, not estimated. Code is in the repo at the end; every kernel self-checks against a reference on every run.

## Setup

- RTX 3060 Laptop: 30 SMs, FP32 peak ~10.9 TFLOPS. WSL2, CUDA 13.3, CUTLASS 4.8.
- M = N = K = 2048, FP32 data, fp32 accumulation unless stated.
- Event timing, cross-checked with nsys; counter work with `ncu --set full`; register counts from `ptxas -v`; instruction mix from `cuobjdump -sass`.

One caveat before someone asks: yes, it's a consumer GPU, and yes, it's one shape. cuBLAS makes its money in heuristic dispatch across shapes; nothing here is a general claim about cuBLAS being beatable. The *methodology* is the transferable part.

The whole ladder:

| Kernel | ms | GFLOPS | vs cuBLAS fp32 |
|---|---|---|---|
| naive | 28.4 | 605 | 9% |
| shared-memory tiled | 27.3 | 630 | 10% |
| register-tiled 8x8 | 3.66 | 4692 | 72% |
| cp.async double-buffered | 3.18 | 5400 | 83% |
| CUTLASS SIMT | 2.34 | 7343 | 113% |
| CUTLASS TF32 tensor cores | 1.77 | 9709 | 149% |
| CUTLASS FP16 tensor cores | 0.88 | 19577 | 300% |
| cuBLAS SGEMM (fp32) | 2.60 | 6600 | 100% |
| cuBLAS GemmEx (fp16 in/out, fp32 acc) | 0.85 | 20288 | 312% |

## Rung 1: naive. I said "memory-bound" and the counters disagreed

Naive: one thread per output element, one global load per FFMA. It lands at 5.5% of peak, and my diagnosis matched everyone's instinct: bad memory behavior.

The counters:

- Compute (SM) SOL: 98.7%
- DRAM SOL: 52.2%
- Top stall: `lg_throttle`, 23.4 cycles per issued instruction

DRAM is half idle. What's saturated is the LSU: every multiply-add ships a global load, the load/store queue fills up, and warps stall waiting to *issue* loads, not waiting on the data behind them. The GPU isn't drowning in bytes; it's drowning in load instructions.

Occupancy here is 99.5%, and it buys nothing. That was the first lesson: "memory-bound" is a conclusion you earn from counters, not a default guess, and occupancy is worthless when the bottleneck is issue bandwidth itself.

## Rung 2: shared-memory tiling -- and almost no speedup

A 32x32 tile staged through shared memory (fixing write-side coalescing along the way) drops DRAM SOL from 52% to 17%. The speedup: 605 to 630 GFLOPS. Basically nothing.

Because each thread still loads its own A and B element from shared memory every k-step. The traffic moved from DRAM to shared, but the instruction shape -- one load per MAC -- didn't change, so the LSU pressure is identical.

This rung is *enabling*, not accelerating. It fixes coalescing and sets up reuse; the reuse only pays once a later rung stops re-reading per step.

## Rung 3: register tiling -- 7.4x in one step

Same structure, but each thread now owns an 8x8 output tile with `acc[8][8]` living entirely in registers. Per k-step that's 16 shared loads (the compiler vectors them into LDS.128) for 64 FFMAs. The loads-per-MAC count drops 64x. That's the entire optimization.

Result: 4692 GFLOPS. The FMA pipe is now the most utilized unit at 42%.

Two details worth dwelling on.

**The 128 registers are the design, not an accident.** The 64-register accumulator plus fragments and addressing lands at exactly 128 registers/thread, and 128 x 256 threads is half the SM's register file -- so only 2 blocks fit per SM and occupancy is 33%. My instinct was to fix the occupancy; the arithmetic says that's wrong. Shrinking the tile to 4x4 doubles occupancy and quadruples loads-per-MAC, trading away the thing that made this rung fast. ncu confirms the limiter: Block Limit Registers = 2 (shared memory would allow 7). Past this point, GEMM means *spending* occupancy to buy arithmetic intensity.

**The top stall is `not_selected`, and that's healthy.** It means when a warp stalls, the scheduler usually has another eligible warp to issue instead. With 15 resident warps per SM and the FMA pipe at 42%, the kernel isn't starved for concurrency -- it's approaching the compute ceiling of the design.

## Rung 4: cp.async double buffering -- +15%, and why it isn't more

Two shared-memory tile sets: while computing tile t, cp.async streams tile t+1. In SASS this shows up as `LDGSTS.E.BYPASS.128` -- global-to-shared without touching registers or the consumer's LSU.

4692 to 5400 GFLOPS, +15%. Not the 2x that blog posts promise, because at this point the kernel is FMA-pipe-bound: prefetching harder can't feed a machine that's already compute-saturated. To jump again, the arithmetic itself has to get cheaper. Enter tensor cores.

## Rung 5: CUTLASS SIMT beats cuBLAS by 11%

CUTLASS 4.8's stock SIMT config -- 128x128x8 tiles, the same shape as mine, no TF32 tricks:

```
mine (register-tile):   3.66 ms,  4692 GFLOPS
CUTLASS SIMT:           2.34 ms,  7343 GFLOPS
cuBLAS SGEMM:           2.60 ms,  6600 GFLOPS
```

Where CUTLASS finds its extra 56% over my kernel: swizzled shared-memory layouts (my `[BK][BN+1]` padding fixes bank conflicts but breaks 16B alignment -- it cost me a real `misaligned address` fault in the cp.async experiment; swizzling solves both at once), larger effective per-thread tiles with better ILP scheduling, and a more mature pipeline.

The takeaway I didn't expect: cuBLAS isn't magic. It's a well-tuned kernel, and at this shape a stock CUTLASS template walks past it. The library's real edge is heuristic dispatch across shapes, which is exactly what a single-shape benchmark doesn't test.

## Rung 6: tensor cores -- 149% and 300%

TF32 (fp32 tensors, 10-bit mantissa in the MMA) and FP16 in/out with fp32 accumulation:

```
CUTLASS TF32: 1.77 ms,  9709 GFLOPS, 149% of cuBLAS fp32
CUTLASS FP16: 0.88 ms, 19577 GFLOPS, 300% of cuBLAS fp32
```

How do you exceed 100% of "peak"? Different peak. The 10.9 TFLOPS FP32 number counts the FFMA pipe; tensor cores are separate hardware, and on GA106 the FP16-with-fp32-accumulate rate is roughly 2x. 19.6 TFLOPS is about 90% of that unit's own ceiling.

The counters here flipped everything the SIMT ladder taught me. The FP16 kernel's most utilized pipe is Tensor at 46.6%, Memory SOL is 39%, DRAM 28% -- nothing is saturated. The bottleneck has moved from "do the math" to "keep the MMA pipe fed": shared-memory turns, barrier stalls, issue gaps between MMAs. This is the entire reason CUTLASS invests in multistage pipelines and warp-specialized producers, and on Hopper, TMA exists largely to continue that fight.

One precision note, because tolerance choices deserve their reasoning attached: at K=2048, per-element error vs the fp32 reference accumulates like sqrt(K) * 2^-11 (~2e-2) for both TF32 and FP16. My first TF32 tolerance was 5e-3 and it failed on real data before I sat down and did the sqrt(K) arithmetic. The failed guess is still in the comment next to the number that works.

## Three bugs I'd have shipped without the harness

Every kernel in the repo verifies against a reference on every run. That discipline caught:

- A merge kernel that read its chunk count from its own `gridDim.y` (1) instead of the partial kernel's (9). Every output was exactly one chunk wrong -- no crash, no NaN, plausibly wrong. The CPU diff caught it because the cache was full of random data.
- A misaligned-address fault from cp.async: the classic `[TILE][TILE+1]` bank-conflict pad breaks 16-byte alignment. Pad to +4 there.
- `cudaFuncSetAttribute` rejecting a request for the device's full shared-memory opt-in: static smem and a per-block reservation come off the top. Ask for what you need.

## What I took away

1. **"Memory-bound" is a conclusion, not a default.** Naive was LSU-issue-bound, register-tiled was FMA-pipe-bound, tensor cores are scheduling-bound. Every rung has a different wall, and counters are the only way to find it.
2. **Occupancy is a currency, not a score.** The naive kernel had 99.5% occupancy and 5% of peak. What matters isn't how many warps are resident -- it's what the resident warps are waiting for.
3. **Libraries are a starting point, not a ceiling.** CUTLASS's stock config beats cuBLAS by 11% at this shape; my hand-written layernorm matches or beats PyTorch's `F.layer_norm` at one shape I benchmarked while the hand-written GEMM only reaches ~63% of `torch.matmul`. Knowing which, and why, is the job.

---

All of it lives in **[github.com/zhangchengzhucufe-dev/cuda-tensor-core-lab](https://github.com/zhangchengzhucufe-dev/cuda-tensor-core-lab)** -- `make run` executes every example with its own verification and timing. The full counter-level write-up is in `docs/gemm_deep_dive.md`, and `docs/interview_qa.md` has the hard-questions version of this material.

Currently looking for GPU kernel / inference-performance work. If your team spends time on any of this, I'd love to talk.
