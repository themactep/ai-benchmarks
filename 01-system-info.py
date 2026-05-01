#!/usr/bin/env python3
"""Detect all AI-capable hardware on this system."""

import os
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


header("SYSTEM OVERVIEW")
print(f"Kernel:   {os.uname().release}")
print(f"Arch:     {os.uname().machine}")
print(f"Hostname: {os.uname().nodename}")

# CPU
header("CPU")
cpuinfo = run("grep 'model name' /proc/cpuinfo | head -1").stdout.strip()
print(f"Model: {cpuinfo.split(':')[1].strip() if ':' in cpuinfo else cpuinfo}")
print(f"Cores:  {os.cpu_count()}")
print(f"Threads: {os.cpu_count()}")

# Frequency info
freq_paths = list(sorted(os.path.join('/sys/devices/system/cpu', d, 'cpufreq/cpuinfo_max_freq')
                          for d in os.listdir('/sys/devices/system/cpu') if d.startswith('cpu')))
if freq_paths:
    freqs = set()
    for fp in freq_paths:
        try:
            with open(fp) as f:
                freqs.add(int(f.read().strip()) // 1000)
        except Exception:
            pass
    if freqs:
        print(f"Frequencies (MHz): {sorted(freqs)}")

# RAM
header("RAM")
mem = run("free -h | grep Mem").stdout.strip()
print(mem)

# P-core vs E-core detection
header("P-CORE / E-CORE DETECTION")
e_cores = []
p_cores = []
for i in range(os.cpu_count()):
    try:
        with open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq") as f:
            freq = int(f.read().strip()) // 1000
        if freq >= 4800:
            p_cores.append(i)
        else:
            e_cores.append(i)
    except Exception:
        break
print(f"P-cores (high freq): {p_cores} → use taskset -c {p_cores[0]}-{p_cores[-1]} for inference")
print(f"E-cores (low freq):  {e_cores}")

# GPU
header("GPU")
gpu_info = run("lspci | grep -iE 'vga|3d|display'").stdout.strip()
print(gpu_info if gpu_info else "No discrete GPU found")

# NVIDIA
nvidia = run("nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null").stdout.strip()
if nvidia and "NVIDIA" in nvidia:
    print(f"\nNVIDIA GPU detected:\n{nvidia}")
else:
    print("NVIDIA GPU: none")

# Intel Arc
header("INTEL ARC / iGPU")
arc = run("lspci -vnn | grep -A 5 '00:02.0' 2>/dev/null").stdout.strip()
if arc:
    print(arc)
else:
    print("No Intel iGPU")

# AMD GPU
header("AMD GPU")
amd = run("lspci -vnn | grep -A 5 'AMD/ATI' 2>/dev/null").stdout.strip()
if amd:
    print(amd)
else:
    print("No AMD GPU")

# NPU
header("NPU (Intel AI Boost)")
npu = run("lspci | grep -iE 'npu|neural|ai.*boost'").stdout.strip()
print(npu if npu else "No NPU detected")

if npu:
    driver = run("lsmod | grep -iE 'intel_vpu|ivpu'").stdout.strip()
    print(f"\nKernel driver: {'LOADED' if driver else 'NOT LOADED'}")
    if driver:
        print(driver)

    # Check device node
    accel = run("ls -la /dev/accel/accel0 2>/dev/null").stdout.strip()
    print(f"\nDevice node:\n{accel}" if accel else "\nNo /dev/accel/accel0")

    # Level Zero NPU
    l0 = run("find /usr/lib -name 'libze_intel_npu*' 2>/dev/null").stdout.strip()
    print(f"\nLevel Zero NPU libs:\n{l0}" if l0 else "\nLevel Zero NPU: NOT INSTALLED")

# CPU instruction sets
header("AI INSTRUCTION SETS")
if os.path.exists("/proc/cpuinfo"):
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("flags"):
                flags = line.strip().split(":")[1].split()
                ai_flags = [f for f in flags if any(x in f for x in ["avx", "fma", "vnni", "amx", "bfloat", "sse"])]
                print(f"AI-related CPU flags: {', '.join(sorted(ai_flags))}")
                break

# OpenVINO devices
header("OPENVINO DEVICES")
try:
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    for dev in devices:
        name = core.get_property(dev, "FULL_DEVICE_NAME")
        print(f"  {dev}: {name}")
except ImportError:
    print("OpenVINO not installed")
except Exception as e:
    print(f"OpenVINO error: {e}")

# PyTorch backends
header("PYTORCH BACKENDS")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:     {torch.cuda.is_available()}")
    print(f"MKL:      {torch.backends.mkl.is_available()}")
    print(f"MKL-DNN:  {torch.backends.mkldnn.is_available()}")

    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX:     {ipex.__version__}")
    except ImportError:
        print("IPEX:     not installed")

    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"XPU:      {torch.xpu.device_count()} device(s)")
        else:
            print("XPU:      not available")
    except Exception:
        print("XPU:      not available")
except ImportError:
    print("PyTorch not installed")

# llama.cpp
header("LLAMA.CPP (Vulkan)")
llama = run("ls ~/llama.cpp/build/bin/llama-cli 2>/dev/null").stdout.strip()
if llama:
    devs = run(f"{llama} --list-devices 2>&1").stdout.strip()
    print(devs)
else:
    print("llama.cpp not built at ~/llama.cpp/build/")

# ONNX Runtime
header("ONNX RUNTIME")
try:
    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Providers: {ort.get_available_providers()}")
except ImportError:
    print("ONNX Runtime not installed")

# CPU governor
header("CPU GOVERNOR")
gov = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null").stdout.strip()
print(f"Current: {gov}")
print("For AI workloads, 'performance' is recommended:")
print("  echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor")

# Disk
header("DISK SPACE")
disk = run("df -h / | tail -1").stdout.strip()
print(disk)

print()
