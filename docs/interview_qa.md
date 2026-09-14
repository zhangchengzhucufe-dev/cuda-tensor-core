# Interview Q&A: the questions this repo should let you survive

Not a textbook. Each answer is the short version you'd actually say,
plus where in the repo the evidence lives. If you can't expand an
answer here into five minutes at a whiteboard, re-read the referenced
file until you can.

## Memory

**Q. Your transpose is 73 -> 190 GB/s. Why does naive transpose lose, and what exactly does the +1 pad fix?**
Naive reads coalesced but *writes* strided: consecutive threads write
consecutive rows, so every 32-float transaction fans out into 32
separate 32B sectors. The tiled version stages through shared memory so
*both* sides coalesce. But a plain `[32][32]` tile then makes the
write-back column access hit one bank: 32-way conflicts, serialized.
`[32][33]` shifts each row one bank over so column j touches 32
distinct banks. Verified with counters: conflict rate ~0.47% of
wavefronts, residue from edge blocks (docs/profiling.md).

**Q. Why is `+1` and not `+4`? Your own cp.async notes say +4.**
+1 float kills bank conflicts for LDS/STS. But cp.async needs 16B
alignment on the shared destination, and a 33-float row stride (132B)
isn't a multiple of 16B -- hence the "misaligned address" fault. There
you pad to +4 instead: `[BK][BN+4]`, so the row stride stays a multiple
of 4 floats = 16B. Same conflict problem, different constraint, so a
different pad. (sgemm_double_buffer.cu header notes)

**Q. Reduction: same FLOPs, 6x apart. Where does the time go?**
Divergence -> control: threads that exited still occupy warp slots in
the first version. Interleaved fixing changes *which* lanes are active
per step. The real win is the last cross-warp step: warp shuffle ops
move values lane-to-lane with no memory, no synchronization, one
instruction. 35.7 -> 227.7 GB/s (reduction.cu).

## GEMM

**Q. Walk me through your GEMM ladder. What's the bottleneck at each rung?**
naive 5.5% of peak, LSU-issue-bound (lg_throttle 23.4 cyc/issue; DRAM
only 52% -- "memory bound" is the wrong diagnosis, it's load-
*instruction* bound). tiled: fixes DRAM (SOL 52->17%) but still 1 load
per MAC, so 5.8%. register-tile: 64 FFMA per 16 shared loads -> FMA
pipe becomes the top unit at 42%, 42.9% of peak. cp.async double
buffer: same occupancy, hides the rest, 49.4%. CUTLASS SIMT: 67%,
beats cuBLAS on this card. Full numbers + counters:
docs/gemm_deep_dive.md.

**Q. Your register-tile kernel runs at 33% occupancy. Isn't that bad?**
It's the design. acc[8][8] = 64 registers of reuse; 128 regs/thread x
256 threads = half the register file = exactly 2 blocks/SM. Halve the
tile and occupancy doubles -- and loads-per-MAC goes up 4x, which the
counters say is the actual bottleneck. Occupancy is currency, not a
score: spend it when it buys arithmetic intensity. (gemm_deep_dive.md,
occupancy table + Block Limit Registers = 2)

**Q. If occupancy is so low, why not more blocks with smaller smem?**
The limiter is registers, not smem: Block Limit Registers = 2 vs
Shared Mem = 7. More blocks/SM would need fewer registers per thread,
which kills the per-thread tile. The only free lunch left is using the
*same* registers better (CUTLASS's ILP scheduling -- which is exactly
where its extra 24 points of peak come from).

**Q. cuBLAS vs CUTLASS -- when do you hand-write?**
Never for stock shapes: CUTLASS beat cuBLAS on my card for SGEMM
(111%). You hand-write (or specialize via CUTLASS templates) when the
op fuses with something -- epilogue (bias+activation), prologue
(dequant), attention masking -- or when the shape is degenerate enough
that the library's heuristics pick badly. cuBLASLt/CUTLASS first,
custom kernel only with a profiler open. (sgemm_cutlass.cu)

**Q. Why does your cp.async double buffer only get +15%? I heard cp.async is a big win.**
Because at BK=8 with 2 stages you hide one 8-wide tile behind ~8 k-steps
of FFMA, and you're already FMA-pipe-bound -- prefetch can't feed a
saturated pipe faster. The win grows when compute:load worsens (BK=16,
3+ stages) or with tensor cores where FFMA cost collapses and loads
re-emerge as the limit. sgemm_double_buffer.cu notes + the SM SOL
47->57% in gemm_deep_dive.md.

**Q. Tensor cores: what actually changes in the code?**
Fragment types (wmma::fragment<matrix_a>...), load_matrix_sync from
shared with correct layout/leading dim, mma_sync in the inner loop,
16x16x16 granularity instead of scalar FFMA. FP16 in / FP32 acc. The
register-tile *structure* (per-thread output tile, shared staging)
carries over unchanged -- WMMA is a change in the arithmetic unit, not
in the tiling idea. (sgemm_wmma.cu)

**Q. TF32 vs FP16 vs FP32 GEMM -- what do you actually give up, in numbers?**
On this card at 2048^3: FP32 SIMT tops at 7.3 TFLOPS (CUTLASS), TF32
runs 9.7, FP16 19.6 -- but each step down the precision ladder widens
the gap to the fp32 reference. TF32 keeps fp32 range and drops the
mantissa to 10 bits; FP16 also narrows range (need loss scaling for
training, less of an issue for inference) and rounds the output. With
K=2048 the per-element error accumulates ~sqrt(K)*2^-11 ~= 2e-2
relative, which is why the verification tolerances in sgemm_cutlass.cu
are per-variant and commented. Knowing *why* a tolerance is 1e-4 for
SIMT and 5e-2 for FP16 is the answer, not the numbers themselves.

**Q. Your FP16 GEMM does 19.6 TFLOPS but FP32 peak is 10.9. How is it above "peak"?**
Two different peaks. FP32 peak counts the FFMA pipe; FP16 tensor cores
are a separate unit -- on GA106 the FP16-TC-FP32acc rate is ~2x the
FP32 pipe (4x with FP16 accumulate). 19.6/21.9 ~= 90% of the tensor
peak, so the FP16 rung is *also* not trivially saturated -- it's
memory/feeding bound, which is exactly why CUTLASS's multistage
pipelining matters more there than bigger tiles.

## NN ops / attention

**Q. Why online softmax instead of regular? What does it save?**
Regular softmax needs all S scores per row resident: materialize SxS,
read it, reduce, read again. Online softmax fuses max and running sum
into the KV loop: keep running m (max) and l (sum), rescale the
accumulator by exp(m_old - m_new) when the max moves. Memory drops from
O(S^2) to O(S x d) -- that IS Flash Attention's core trick.
(attention_forward.cu)

**Q. What does a KV cache actually change, and what does it cost?**
Decode generates one token per step, but every past K/V is needed
again -- so you cache them: append the new K/V row to the cache, attend
with one query row over T cached tokens. Cost per step is O(T x H x D)
memory -- linear in context, no S x S scores at all. The catch my file
demonstrates: the obvious one-block-per-head kernel crawls at 25 GB/s
because 8 blocks can't hide DRAM latency, and splitting the T loop
across blocks with per-chunk max/sum combining (flash-decoding) takes
it to 312 GB/s at a 16K cache -- basically the card's read bandwidth.
Same math, 11x apart, both verified. (decode_kv_cache.cu)

**Q. Why paged attention? (the follow-up you'll get)**
Contiguous caches sized for max_seq_len waste memory: batch x max_len
reservation with ragged real usage. Paged KV cache stores it in fixed
blocks with a block table (virtual memory for caches): ~no fragmentation,
arbitrary lengths, and the kernel indexes through the table per K/V
tile. Same idea as OS pages; the payoffs are batching efficiency, not
kernel FLOPs.

**Q. Your layernorm uses E[x^2] - mean^2. Every numerics guide says avoid that. Why is it okay here?**
The catastrophic-cancellation warning applies when variance is tiny
relative to the mean. With normalized inputs and activations in
[-10, 10]-ish, the subtraction loses ~1-2 ulps -- and the whole thing
is verified against a double-precision CPU pass, so if it drifted, the
harness would say so. Welford is the right call for extreme dynamic
ranges; here the two-pass structure would cost an extra block
reduction for accuracy you can't measure. (layernorm.cu,
transformer_block.cu for the double-precision check)

**Q. Bias+GELU fusion: 2.0 -> 0.64 ms. Why 2x and not more?**
The unfused version does one extra full read+write of the intermediate
-- 3 passes of memory vs 1. But it stays bandwidth-bound either way, and
the fused kernel still has to read A and write C once, so the floor is
~1/3 of the unfused time minus launch overhead; 0.64/2.0 ≈ 0.32 is
right at that floor. Fusion can't beat memory bandwidth, it just stops
paying it twice. (bias_gelu_fused.cu)

## Streams / graphs

**Q. Multi-stream gave you only 1.03x. Why? Would it help on A100?**
Consumer chips have one copy engine: H2D copies serialize regardless of
stream count (nsys shows all 16 transfers on one engine). A100/H100 have
multiple engines + NVLink, so the same code actually overlaps copy with
compute. So it's not "streams don't work" -- streams expose parallelism
and the hardware decides how much you get. (07_streams,
docs/profiling.md)

**Q. When do CUDA graphs actually pay?**
When the per-launch CPU cost is a meaningful fraction of the frame:
tiny kernels, many of them, short queue. 40-70% on a 3-tiny-kernel
toy pipeline, but only 1.07x on the transformer block where kernels
have real work -- the GPU was already the bottleneck. Graphs remove
*submit* cost, not kernel time. Measure before you reach for them.
(13_graphs, 14_transformer_block)

## Method (the questions senior interviewers actually ask)

**Q. How do you know any of your numbers are real?**
Same data, CPU or cuBLAS reference, compare_close with relative tol;
best-of-N event timing; nsys cross-check showing event-timing spread is
~0.06%; ncu numbers quoted with the caveat that it locks clocks.
Every binary self-verifies on every run -- including the transformer
block case where the CPU reference itself was wrong and the GPU was
right. That one taught me to verify the verifier.

**Q. What did you get wrong along the way?**
Have real answers ready: the `__shfl_sync` broadcast that doesn't cross
warps; inclusive-vs-exclusive scan of block sums; a grid-size bug that
silently biased only the Q slice (underlaunch doesn't crash); the cp.async
alignment fault from the +1 pad; a CPU reference bug where the GPU
was right; and from the decode kernel: the merge kernel that read its
chunk count from `gridDim.y` of its own launch instead of the partial
kernel's -- every output was exactly one chunk wrong, which only the
CPU diff caught, and only because the test appended to the cache *after*
filling it with random data. Also: requesting the device's full smem
opt-in byte-for-byte fails ("invalid argument") because static smem and
the per-block reservation come off the top -- ask for what you need.
README lists more -- being able to narrate a debugging story
with evidence beats any list of buzzwords.

**Q. What would you do next with unlimited time?**
BK=16 / 3-stage cp.async pipeline; a fused-epilogue CUTLASS GEMM
(bias+GELU in the TF32/FP16 kernel); a paged cache layout + GQA on top
of the split decode kernel; wiring the decode path + graph replay into
the torch extension for a full autoregressive loop; the same ladder on
Hopper ideas (TMA, warp specialization).

## Wrapping kernels for a framework

**Q. What does it take to hand a lab kernel to PyTorch?**
cpp_extension/pybind for binding; ATen tensor plumbing (contiguity,
dtype, device guards); TORCH_CHECK contracts at the boundary instead of
printf-and-exit; and correctness against torch's own op, not against a
private harness. One honest surprise from doing it: the register-tile
GEMM lands at ~63% of torch.matmul (torch calls cuBLAS), but the
layernorm wrapper runs even with F.layer_norm at 8192x512 and beats it
on a good run -- the library doesn't win every shape, which is the
whole justification for custom kernels existing. (15_torch_extension)
