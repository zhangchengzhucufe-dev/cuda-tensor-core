# The GEMM ladder, profiled: naive → tiled → register-tile → cp.async → CUTLASS → cuBLAS

Every number below is measured on the card this repo was developed on
(RTX 3060 Laptop, 30 SMs @ 1.43 GHz, FP32 peak **10.94 TFLOPS**), at a
uniform **M=N=K=2048**, with `ncu --set full` (2026-09-13, CUDA 13.3).
This is the counter-level analysis docs/profiling.md used to say was
impossible on WSL2 -- see the update there for why it works now.

One caveat on cross-source numbers: `ncu` locks clocks to base during
replay, so its durations run slightly slower than the event timing the
binaries print (naive: 28.39 ms by event vs 29.33 ms under ncu). The
GFLOPS column below is from the binaries' own event timing; the SOL /
occupancy / stall columns are from ncu. Both are honest, they just
aren't the same clock.

## The ladder

| kernel | ms | GFLOPS | vs cuBLAS fp32 | regs/thr | smem/block | theo occ | ach occ | warps/SM |
|---|---|---|---|---|---|---|---|---|
| naive | 28.39 | 605 | 9.3% | 40 | 0 KB | 100% | 99.5% | 47.8 |
| tiled (32x32) | 27.25 | 630 | 9.7% | 38 | 8 KB | 66.7% | 66.6% | 32.0 |
| register-tile 8x8 | 3.66 | 4692 | 72% | 128 | 8 KB | 33.3% | 31.9% | 15.3 |
| cp.async double-buffer | 3.18 | 5400 | 83% | 123 | 16.6 KB | 33.3% | 32.0% | 15.4 |
| CUTLASS SIMT 128x128x8 | 2.34 | 7343 | 113% | -- | -- | -- | -- | -- |
| CUTLASS TF32 TC 128x128x32 | 1.77 | 9709 | 149% | -- | 96 KB | -- | -- | -- |
| CUTLASS FP16 TC 128x128x64 | 0.88 | 19577 | 300% | -- | 96 KB | -- | -- | -- |
| cuBLAS SGEMM (fp32) | 2.61 | 6600 | 100% | -- | -- | -- | -- | -- |
| cuBLAS GemmEx (fp16/fp32acc) | 0.85 | 20288 | 312% | -- | -- | -- | -- | -- |

(ptxas -v for the register/smem columns; `sgemm_cutlass` prints the full
table and cross-checks every variant against the fp32 reference at a
per-variant tolerance. Two more peaks for the roofline math: TF32 tensor
peak on this card is 10.94 TFLOPS -- half the FP16 rate -- and FP16
with FP32 accumulate is ~21.9 TFLOPS, so the FP16 rungs sit at ~89-93%
of *their own* peak.)

## Rung 0 -> 1: naive. It's not DRAM-bound, which surprised me

The obvious story is "naive GEMM is slow because it's memory bound".
The counters say something more specific:

| | value |
|---|---|
| Compute (SM) SOL | 98.7% |
| L1TEX (mem SOL) | 98.7% |
| DRAM SOL | 52.2% |
| top stall: `lg_throttle` | **23.4 cycles/issue** |
| second stall: `long_scoreboard` | 8.75 |

DRAM sits at half utilization. What's saturated is the LSU/L1 path:
naive issues **one global load per FFMA** (2 GFLOP needs 2 G loads), and
`lg_throttle` at 23.4 cycles/issue means the load/store unit queue is
full -- warps are waiting to *issue loads*, not waiting on DRAM behind
them. The card is drowning in load instructions, not in bytes. That's
why the naive kernel gets almost nothing from its 99.5% occupancy:
47.8 resident warps per SM and every one of them wants the LSU.

Also worth having ready for the "just raise occupancy" suggestion: the
naive kernel here has near-perfect occupancy and near-worst throughput.
Occupancy only helps if it buys latency hiding for the bottleneck
resource, and here the bottleneck is issue bandwidth itself.

## Rung 1 -> 2: shared-memory tiling. Coalescing fixes DRAM, but the loads-per-MAC problem stays

`tiled` (32x32 block tile, one output per thread, [TILE][TILE+1] smem):
DRAM SOL drops 52.2% -> 16.8%, but GFLOPS barely move (605 -> 630).

Why: each thread still loads its own A element and B element *per k
step* from shared memory and does one FFMA on them. The instruction mix
is still ~1 LDS + 1 LDS + 1 FFMA per MAC. The traffic moved from global
to shared, but the LSU still touches a load port for every FLOP, so the
kernel stays memory-pipe-bound (mem SOL 79.6%, lg/short-scoreboard
dominated). Shared memory tiling is the *enabling* rung -- it fixes
coalescing and sets up reuse -- but the reuse only pays when a later
rung stops re-reading per step.

It also costs occupancy for nothing yet: the 32x32-thread block plus 8 KB
of smem lands every block limit (registers, smem, warps) at exactly
1 block/SM = 32 of 48 warp slots = 66.7% theoretical occupancy. Fine --
occupancy wasn't the bottleneck, as rung 0 already proved.

## Rung 2 -> 3: register tiling. Give each thread more output, get 7.4x

`register-tile`: each thread computes an 8x8 output tile (TM=TN=8),
holding `acc[8][8]` in registers. Per k step a thread does 16 shared
loads (8 A frags + 8 B frags, and the compiler vectorizes them into
LDS.128) and **64 FFMAs**. The loads-per-MAC count drops 64x. That's
the rung, in one number.

| | value |
|---|---|
| FMA pipe utilization | 42-44% ("FMA Lite" sub-pipe, highest pipe) |
| mem SOL / DRAM SOL | 76.8% / 14.6% |
| top "stall" | `not_selected` 1.77 -- a *good* sign |
| regs/thread | 128 (0 spills) |
| occupancy | 33.3% theoretical, 31.9% achieved |

`not_selected` being the top stall means: when a warp stalls, the
scheduler usually has another eligible warp to issue instead -- the SM
rarely goes idle. With only 15.3 warps/SM, the FMA pipe is at 42% not
because warps wait on memory (long_scoreboard is only 0.93) but because
there aren't enough resident FFMA streams to fill the pipe. Which
predicts the next lever.

The 128 registers are not an accident: acc[8][8] = 64 registers of
state, plus fragments and addressing. 128 regs x 256 threads = 32K
registers = exactly half the SM's 64K file, so **2 blocks/SM is the
register-imposed ceiling** (ncu confirms: Block Limit Registers = 2,
vs 7 by smem). docs/profiling.md used to guess "6 blocks/SM" from
literature numbers for a smaller tile; the real measurement says 2.
Halving TM/TN to 4x4 (acc = 16 regs) would roughly double occupancy --
and lose the reuse. This is the file where it finally clicked for me
that a GEMM kernel is a trade: you give up occupancy to get arithmetic
intensity.

## Rung 3 -> 4: cp.async double buffering. Hide what's left

`double-buffer`: two smem tile sets; while computing tile t, cp.async
pulls tile t+1. In SASS, cp.async is `LDGSTS.E.BYPASS.128` (4 sites:
A+B tiles x 2 buffers) -- data goes global->shared without landing in
registers or occupying the consumer's LSU slot.

| | regtile | double-buffer |
|---|---|---|
| event ms (2048^3) | 3.66 | 3.18 |
| GFLOPS | 4692 | 5400 |
| SM SOL | 47.0% | 57.1% |
| mem SOL | 76.8% | 64.6% |

+15% for ~10 lines of changes. Both run at the same 33% occupancy, but
the double-buffered version's FMA pipe stays busier (57% vs 47%): the
k-step's dead phase is gone. The file's notes put the win at +15% and
explain why it stops there -- consistent with the counters: we're
FMA-pipe-bound now, so the remaining gain was never going to come from
prefetching harder. (The notes in sgemm_double_buffer.cu on BK=16 /
3-stage pipelines are the right next experiments.)

## Rung 4: CUTLASS -- what a production template does that I didn't

`sgemm_cutlass.cu` benchmarks CUTLASS 4.8's stock SIMT config (same
128x128x8 tile shape, same OpClassSimt, no TF32) against both:

```
mine (register-tile):    3.659 ms,   4695.6 GFLOPS,  71.2% of cuBLAS
CUTLASS (simt 128x128):  2.340 ms,   7342.6 GFLOPS, 111.3% of cuBLAS
cuBLAS                  : 2.604 ms,   6597.9 GFLOPS
```

CUTLASS's default SIMT kernel beats cublasSgemm by 11% on this card
(cuBLAS's SGEMM is clearly leaving something on the table for this
shape; no TF32 involved on either side). What it has that my kernel
doesn't:

- **swizzled shared memory layout**: my `Bs[BK][BN+1]` pad fixes bank
  conflicts; CUTLASS permutes tile coordinates so conflicts vanish and
  vectorization survives at the same time (the pad trick breaks 16B
  alignment -- my own cp.async notes hit exactly that)
- **larger per-thread tiles**: more like 4x8x8 macro-Fragment structure
  with better ILP scheduling -- more FFMA in flight per warp without
  more registers per result
- **ping-pong + multistage machinery** that generalizes (the same
  template instantiates the tensor-core versions)

The short version: my kernel reaches 42.9% of FP32 peak with
pedagogically-clean code; CUTLASS reaches 67.1% with the same tile
shape by eating the complexity in layout policy and scheduling rather
than in bigger tiles. That gap (43% -> 67%) is precisely the part of
kernel engineering that is "just engineering" -- and it's why you use
CUTLASS as the substrate for anything real.

## SASS: the two kernels side by side

Static instruction counts of the inner (fully unrolled) section
(`cuobjdump -sass` on the built binaries):

| opcode | register-tile | double-buffer | what it means |
|---|---|---|---|
| FFMA | 512 | 512 | 8 k-steps x 64 MACs, fully unrolled both |
| LDS | 38 (mostly LDS.128) | 32 | frag loads, vectorized 4-wide |
| LDG | 2 | 6 | dbuf prefetches A and B for next tile |
| STS | 5 | 0 | dbuf's global->shared is cp.async, no STS |
| LDGSTS.E.BYPASS.128 | 0 | 4 | the cp.async itself |
| BAR.SYNC | 2 | 2 | one per k-step loop body |

The FFMA:LDS ratio is the arithmetic-intensity story in one glance:
512 FFMAs against ~38 load instructions (each moving 4 floats), i.e.
~3.4 FLOPs per shared-memory word. The naive kernel is the same
computation at 1 FFMA per LDG.

## Rung 5: tensor cores -- collapse the arithmetic, inherit a new bottleneck

Same file, two more CUTLASS configs (`OpClassTensorOp`): TF32 (float
in/out, 10-bit mantissa in the MMA) and FP16 in/out with FP32
accumulate. Shapes pinned explicitly -- TF32's mma is 16x8x8, FP16's is
16x8x16, and the template defaults don't resolve to either on their
own (the compile dies with an incomplete `Mma` type, which is how I
learned these configs name an actual hardware instruction, not a
tuning knob).

```
CUTLASS (tf32 tensor) :  1.769 ms,   9709.0 GFLOPS,  149.0% of cuBLAS fp32
CUTLASS (fp16 tensor) :  0.878 ms,  19577.4 GFLOPS,  300.5% of cuBLAS fp32
cuBLAS  (fp16 tensor) :  0.847 ms,  20288.4 GFLOPS
```

Both land within ~90-93% of their own precision's peak (TF32 peak =
10.94, FP16/FP32acc peak ~= 21.9 TFLOPS on GA106). But the counters say
neither is *pipe-saturated* -- the FP16 kernel's highest pipe is
Tensor (FP) at **46.6%**, with Memory SOL ~39% and DRAM only ~28%:

```
ampere_fp16_s1688gemm_fp16_256x64... : SM SOL 46.6%, Mem SOL 39.0%, DRAM 27.7%
ampere_sgemm_128x64_nn (cuBLAS fp32): SM SOL 70.3%, Mem SOL 53.6%, DRAM 23.3%
```

Nothing is anywhere near 90%, so no single unit is the wall -- the
time is going into the gaps *between* the work: shared-memory turns,
barrier stalls, MMA issue gaps. Everything the SIMT ladder taught
flips here: tensor cores made the arithmetic nearly free, so the game
is no longer "raise arithmetic intensity" but "keep the MMA pipe
fed" -- which is what CUTLASS's multistage cp.async pipelines, warp
tiling policies and (on Hopper) warp-specialized producers are all
for. 46.6% pipe utilization but ~90% of peak achieved means the
scheduling is overlapping the gaps well already; whatever's left of
the last 10% is the genuinely hard part.

One precision footnote: per-element error vs the fp32 reference
accumulates like sqrt(K) * 2^-11 for both TF32 and FP16 (same 10-bit
mantissa) ~= 2e-2 at K=2048. So sgemm_cutlass.cu verifies each variant
at its own commented tolerance (1e-4 SIMT, 5e-2 TF32/FP16). I picked
5e-3 for TF32 first and it failed on real data before I sat down and
did the sqrt(K) arithmetic -- the failed guess is still in the comment
next to the number that works.

## What the library looks like from inside

cuBLAS's fp32 SGEMM kernel on this shape is `ampere_sgemm_128x64_nn`:
SM SOL 70%, and it lands at 6.6 TFLOPS -- between my register-tile
kernel and CUTLASS SIMT. "cuBLAS" is not magic; it is a well-tuned
CUTLASS-shaped kernel, and `sgemm_cutlass.cu` shows a stock CUTLASS
config beating it by 11% at this size. The library's real edge is
heuristics across shapes, not a single-shape miracle.

## Roofline sanity check

Arithmetic intensity of GEMM at 2048^3 is huge (2MNK FLOPs vs 3MNK x 4B
moved if you re-read everything, ~342 FLOP/byte even un-tiled) -- the
problem is compute-bound on paper for *any* decent tiling. The measured
barrier is therefore never DRAM (and indeed DRAM SOL never exceeds
52%, and that's the naive kernel), it's the instruction pipes: first
LSU issue bandwidth (naive/tiled), then the FMA pipe (regtile/dbuf).
"Memory bound vs compute bound" is a statement about *which unit you
saturate*, and the ladder moves you from the first to the second.

## What's still on the table

- BK=16 or a 3-stage cp.async pipeline for the hand-written path
- a fused epilogue (bias+GELU inside the TF32/FP16 GEMM) -- the
  production answer to what 09_nn_ops does by hand; the wrapper for
  consuming it from a real framework already exists in
  15_torch_extension
- split-K for skinny matrices (M,N >> K) where occupancy leaves SMs
  idle at small grid sizes
- Hopper-era answers to the same problem (TMA + warp-specialized
  producers/consumers) -- same ideas, different hardware units
