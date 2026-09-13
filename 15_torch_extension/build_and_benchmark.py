# Build the extension, prove it correct against torch, then race it.
#
#   python3 build_and_benchmark.py
#
# First run compiles ext.cu with torch's cpp_extension (needs a CUDA build
# of torch + nvcc; takes a minute or two, cached after that).
# Everything runs on whatever GPU torch sees.
import time

import torch
from torch.utils.cpp_extension import load

mod = load(
    name="lab_ops",
    sources=["ext.cu"],
    verbose=False,
)
dev = "cuda"
torch.manual_seed(0)

# ---- correctness: same contract as every standalone example in the repo --
a = torch.randn(2048, 2048, device=dev)
b = torch.randn(2048, 2048, device=dev)
mine = mod.mm(a, b)
ref = a @ b
torch.testing.assert_close(mine, ref, rtol=1e-4, atol=1e-4)
print("mm vs torch.matmul: allclose (rtol=atol=1e-4)")

x = torch.randn(4096, 512, device=dev)
w = torch.randn(512, device=dev)
bb = torch.randn(512, device=dev)
mine_ln = mod.layernorm(x, w, bb)
ref_ln = torch.nn.functional.layer_norm(x, (512,), w, bb, 1e-5)
torch.testing.assert_close(mine_ln, ref_ln, rtol=1e-4, atol=1e-5)
print("layernorm vs F.layer_norm: allclose (rtol=1e-4, atol=1e-5)")

# and the loud failure the wrapper promises when you break the tile contract
try:
    mod.mm(a[:100], b[:100])
    print("ERROR: expected the divisibility TORCH_CHECK to fire")
except RuntimeError as e:
    print(f"mm on 100x100 correctly rejected: {str(e).splitlines()[0][:60]}...")

# ---- benchmark: the honest question is not 'is it fast' but 'vs what' ----
def bench(fn, reps=50):
    for _ in range(5):
        fn()  # warm up clocks and caches
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms

ms_mine, ms_torch = bench(lambda: mod.mm(a, b)), bench(lambda: a @ b)
gflop = 2 * 2048**3 / 1e9
print(f"\nGEMM 2048^3:  mine {ms_mine:.3f} ms ({gflop/ms_mine*1e3:.0f} GFLOPS)"
      f"  torch {ms_torch:.3f} ms ({gflop/ms_torch*1e3:.0f} GFLOPS)"
      f"  -> {ms_torch/ms_mine*100:.1f}% of torch")

x2 = torch.randn(8192, 512, device=dev)
ms_mine, ms_torch = bench(lambda: mod.layernorm(x2, w, bb)), bench(
    lambda: torch.nn.functional.layer_norm(x2, (512,), w, bb, 1e-5))
print(f"LN 8192x512:  mine {ms_mine:.4f} ms  torch {ms_torch:.4f} ms"
      f"  -> {ms_torch/ms_mine*100:.1f}% of torch")
