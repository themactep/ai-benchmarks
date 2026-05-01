#!/usr/bin/env python3
"""
OpenVINO benchmarks: CPU vs GPU vs NPU.
Requires: openvino, numpy
Run: python3 03-openvino-all.py [--device CPU|GPU|NPU|ALL]
"""

import openvino as ov
import numpy as np
import time
import sys
import argparse


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def build_matmul_model(size=4096, dtype=np.float32):
    a = ov.opset13.parameter([size, size], ov.Type.f32, name="a")
    b = ov.opset13.parameter([size, size], ov.Type.f32, name="b")
    mm = ov.opset1.matmul(a, b, False, False)
    result = ov.opset13.result(mm, name="output")
    return ov.Model([result], [a, b], "matmul")


def build_cnn_model():
    B, C, H, W = 1, 3, 224, 224
    param = ov.opset13.parameter([B, C, H, W], ov.Type.f32, name="input")

    x = ov.opset13.convolution(
        param,
        ov.opset13.constant(np.random.randn(32, 3, 5, 5).astype(np.float32)),
        strides=[2, 2], pads_begin=[2, 2], pads_end=[2, 2], dilations=[1, 1],
    )
    x = ov.opset13.relu(x)
    x = ov.opset13.convolution(
        x,
        ov.opset13.constant(np.random.randn(64, 32, 3, 3).astype(np.float32)),
        strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1],
    )
    x = ov.opset13.relu(x)
    x = ov.opset13.convolution(
        x,
        ov.opset13.constant(np.random.randn(128, 64, 3, 3).astype(np.float32)),
        strides=[2, 2], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1],
    )
    x = ov.opset13.relu(x)
    x = ov.opset13.reshape(x, [1, -1], False)
    shape_after = 128 * 56 * 56
    x = ov.opset1.matmul(
        x,
        ov.opset13.constant(np.random.randn(shape_after, 1000).astype(np.float32)),
        False, False,
    )
    result = ov.opset13.result(x, name="output")
    return ov.Model([result], [param], "cnn")


def benchmark_device(core, model, device, input_data, warmup=3, iterations=20):
    compiled = core.compile_model(model, device)
    infer = compiled.create_infer_request()

    for _ in range(warmup):
        infer.infer(input_data)

    start = time.time()
    for _ in range(iterations):
        infer.infer(input_data)
    elapsed = time.time() - start

    ms = elapsed * 1000 / iterations
    return ms, elapsed


# --- Main ---
parser = argparse.ArgumentParser(description="OpenVINO AI benchmarks")
parser.add_argument("--device", default="ALL",
                    choices=["CPU", "GPU", "NPU", "ALL"],
                    help="Device to benchmark (default: ALL)")
parser.add_argument("--matmul-size", type=int, default=4096,
                    help="Matrix size for matmul benchmark (default: 4096)")
args = parser.parse_args()

core = ov.Core()

# Device detection
header("DEVICE DETECTION")
print(f"Available: {core.available_devices}")
for dev in core.available_devices:
    name = core.get_property(dev, "FULL_DEVICE_NAME")
    print(f"  {dev}: {name}")

devices = core.available_devices if args.device == "ALL" else [args.device]
devices = [d for d in devices if d in core.available_devices]
if not devices:
    print("No matching devices available.")
    sys.exit(1)

# ----------------------------------------------------------------
# MATMUL BENCHMARK
# ----------------------------------------------------------------
header(f"MATMUL ({args.matmul_size}×{args.matmul_size}, float32)")
size = args.matmul_size
model = build_matmul_model(size)
a_data = np.random.randn(size, size).astype(np.float32)
b_data = np.random.randn(size, size).astype(np.float32)
input_data = [a_data, b_data]

results = {}
for device in devices:
    try:
        ms, elapsed = benchmark_device(core, model, device, input_data, iterations=20)
        flops = 2 * size**3 * 20
        tflops = flops / elapsed / 1e12
        results[device] = {"ms": ms, "tflops": tflops}
        print(f"  {device:>5s}: {ms:7.1f} ms/inf = {tflops:.2f} TFLOPS")
    except Exception as e:
        print(f"  {device:>5s}: ERROR - {str(e)[:120]}")

# Speedup
if len(results) >= 2 and "CPU" in results:
    cpu_tflops = results["CPU"]["tflops"]
    print()
    for dev, r in results.items():
        if dev != "CPU":
            print(f"  {dev} vs CPU: {r['tflops']/cpu_tflops:.1f}x faster")

# ----------------------------------------------------------------
# CNN BENCHMARK
# ----------------------------------------------------------------
header("CNN BENCHMARK (ConvNet, float32)")
model_cnn = build_cnn_model()
input_cnn = np.random.randn(1, 3, 224, 224).astype(np.float32)

for device in devices:
    try:
        ms, elapsed = benchmark_device(core, model_cnn, device, [input_cnn], iterations=50)
        fps = 50 / elapsed
        print(f"  {device:>5s}: {ms:7.1f} ms/inf = {fps:5.0f} FPS")
    except Exception as e:
        print(f"  {device:>5s}: ERROR - {str(e)[:120]}")

print()
print("Done.")
