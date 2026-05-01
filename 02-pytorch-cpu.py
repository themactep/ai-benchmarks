#!/usr/bin/env python3
"""
PyTorch CPU AI benchmarks with optional Intel IPEX acceleration.
Run: python3 02-pytorch-cpu.py
"""

import torch
import time
import sys


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# Check IPEX
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
    ipex_version = ipex.__version__
except ImportError:
    has_ipex = False
    ipex_version = None

header("PYTORCH STATUS")
print(f"PyTorch:  {torch.__version__}")
print(f"IPEX:     {ipex_version if has_ipex else 'not installed'}")
print(f"MKL:      {torch.backends.mkl.is_available()}")
print(f"MKL-DNN:  {torch.backends.mkldnn.is_available()}")
print(f"Threads default: {torch.get_num_threads()}")

# ----------------------------------------------------------------
# 1. MATRIX MULTIPLICATION (throughput)
# ----------------------------------------------------------------
header("MATRIX MULTIPLICATION (2048×2048)")

for dtype in [torch.float32, torch.bfloat16]:
    try:
        # Use tiny matrix for bfloat16 (no AMX on client CPUs = software emulation, extremely slow)
        size = 2048 if dtype == torch.float32 else 64
        iterations = 5 if dtype == torch.float32 else 3
        a = torch.randn(size, size, dtype=dtype)
        b = torch.randn(size, size, dtype=dtype)

        # Warmup
        _ = torch.mm(a, b)

        start = time.time()
        for _ in range(iterations):
            _ = torch.mm(a, b)
        elapsed = time.time() - start

        flops = 2 * size**3 * iterations
        tflops = flops / elapsed / 1e12
        ms = elapsed * 1000 / iterations
        if dtype == torch.bfloat16 and tflops < 0.001:
            print(f"  {str(dtype):>15s}: {ms:7.1f} ms (64×64) = {tflops*1e6:.1f} MFLOPS  ⚠ NO AMX — avoid bfloat16 on this CPU")
        else:
            print(f"  {str(dtype):>15s}: {ms:7.1f} ms = {tflops:.2f} TFLOPS")
    except Exception as e:
        print(f"  {str(dtype):>15s}: ERROR - {e}")


# ----------------------------------------------------------------
# 2. THREAD SCALING
# ----------------------------------------------------------------
header("THREAD SCALING (1024×1024, float32)")

best_threads = 1
best_gflops = 0
for threads in [1, 4, 8, 16, 22]:
    torch.set_num_threads(threads)
    size = 1024
    a = torch.randn(size, size, dtype=torch.float32)
    b = torch.randn(size, size, dtype=torch.float32)

    _ = torch.mm(a, b)
    start = time.time()
    for _ in range(10):
        _ = torch.mm(a, b)
    elapsed = time.time() - start

    gflops = 2 * size**3 * 10 / elapsed / 1e9
    marker = " *" if gflops > best_gflops else ""
    if gflops > best_gflops:
        best_gflops = gflops
        best_threads = threads
    print(f"  {threads:2d} threads: {elapsed:.3f}s = {gflops:7.1f} GFLOPS{marker}")

torch.set_num_threads(best_threads)
print(f"\n  Best: {best_threads} threads → {best_gflops:.1f} GFLOPS")
print(f"  Recommendation: torch.set_num_threads({best_threads})")


# ----------------------------------------------------------------
# 3. TRANSFORMER LAYER SIMULATION
# ----------------------------------------------------------------
header("TRANSFORMER LAYER SIMULATION (4K hidden, 32 heads)")

def benchmark_transformer(dtype, n_layers=20, batch=1, seq=64, hidden=4096, heads=32, head_dim=128):
    x = torch.randn(batch, seq, hidden, dtype=dtype)
    w_qkv = torch.randn(hidden, 3 * hidden, dtype=dtype)
    w2 = torch.randn(hidden, hidden * 4, dtype=dtype)
    w3 = torch.randn(hidden * 4, hidden, dtype=dtype)

    def layer(x):
        qkv = torch.matmul(x, w_qkv)
        q, k, v = qkv.split(hidden, dim=-1)
        q = q.view(batch, seq, heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        # bfloat16 softmax is slow — upcast to float32, softmax, downcast back
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, hidden)
        return torch.matmul(torch.relu(torch.matmul(out, w2)), w3)

    _ = layer(x)

    start = time.time()
    for _ in range(n_layers):
        x = layer(x)
    elapsed = time.time() - start

    tps = seq * n_layers / elapsed
    return elapsed, tps


# Float32 transformer (primary benchmark)
elapsed, tps = benchmark_transformer(torch.float32)
print(f"  torch.float32: {elapsed:.2f}s for 20 layers → ~{tps:.0f} tok/s")

# bfloat16 transformer (quick check — very slow without AMX)
print(f"  torch.bfloat16: SKIPPED (no AMX on this CPU — would take several minutes)")


# ----------------------------------------------------------------
# 4. MEMORY BANDWIDTH (LLM bottleneck)
# ----------------------------------------------------------------
header("MEMORY BANDWIDTH (float32, 64M elements)")

nelem = 64 * 1024 * 1024
x = torch.randn(nelem, dtype=torch.float32)
y = torch.randn(nelem, dtype=torch.float32)

_ = x + y
start = time.time()
for _ in range(10):
    z = x + y
elapsed = time.time() - start

bytes_moved = nelem * 4 * 3 * 10  # read x, read y, write z
bw = bytes_moved / elapsed / 1e9
print(f"  {elapsed:.3f}s, {bw:.1f} GB/s")



print()
print("Done.")
print(f"Recommended: torch.set_num_threads({best_threads})")
print()
print("NOTE: bfloat16 is not hardware-accelerated on this CPU (no AMX).")
print("      Use float32 for training or INT8 quantization for inference.")
print("      For bfloat16/fp16 inference, use OpenVINO on the GPU or NPU.")
