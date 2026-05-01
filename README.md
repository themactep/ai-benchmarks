# AI Benchmarks

Benchmark suite for the Intel Core Ultra 9 185H (Meteor Lake) AI PC.

**Accelerators**: CPU (AVX2 + VNNI), Intel Arc iGPU, Intel AI Boost NPU (11 TOPS INT8), AMD Radeon RX 580 (Vulkan)

## Quick start

```bash
# Run everything
./run-all.sh

# Or individually:
python3 01-system-info.py       # Hardware detection
python3 02-pytorch-cpu.py       # PyTorch CPU benchmarks
python3 03-openvino-all.py      # OpenVINO CPU / GPU / NPU
bash 04-llama-cpp-check.sh      # llama.cpp Vulkan devices
```

Run a specific device only:
```bash
python3 03-openvino-all.py --device NPU
python3 03-openvino-all.py --device GPU
python3 03-openvino-all.py --matmul-size 2048
```

## Expected results (this system)

| Device | Matmul 4096² f32 | CNN FPS | Best for |
|--------|-----------------|---------|----------|
| CPU    | ~0.2 TFLOPS     | ~32     | Training, float32 inference |
| GPU    | ~0.7 TFLOPS     | ~65     | Vision models, OpenVINO |
| NPU    | ~1.2 TFLOPS     | ~58     | Low-power INT8 inference |

**PyTorch CPU** (with IPEX + perf governor): ~0.57 TFLOPS matmul, ~493 GFLOPS peak at 16 threads.

**llama.cpp Vulkan LLM inference**:
- Intel Arc: 48 GB shared → run 70B models
- AMD RX 580: 8 GB VRAM → 7B-13B models at ~20-40 tok/s

## Prerequisites

All packages are already installed. If setting up from scratch:

```bash
# Core AI libraries
pip install torch intel-extension-for-pytorch openvino numpy

# Intel GPU compute (OpenVINO GPU plugin needs this)
sudo apt install intel-opencl-icd libze-intel-gpu1

# Intel NPU (downloaded from GitHub)
# 1. Level Zero NPU driver:
#    https://github.com/intel/linux-npu-driver/releases
#    → intel-level-zero-npu_*.deb
# 2. NPU compiler:
#    https://github.com/openvinotoolkit/npu_compiler/releases
#    → extract lib → copy to openvino/libs/libopenvino_intel_npu_compiler.so

# CPU governor (already set to performance)
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor

# User must be in render group for NPU access
sudo usermod -a -G render $USER
# Log out and back in for this to take effect

# llama.cpp with Vulkan
cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

## Architecture notes

- **CPU**: 6 P-cores (4.8-5.1 GHz) + 8 E-cores (2.5-3.8 GHz) + 2 LP E-cores. No AMX. Has AVX-VNNI for INT8 acceleration. Pin PyTorch to P-cores with `torch.set_num_threads(8)` or `taskset -c 0-11`.

- **GPU (Arc)**: Integrated GPU sharing system RAM (48 GB max). Access via OpenVINO GPU plugin or llama.cpp Vulkan backend.

- **NPU (AI Boost)**: Dedicated AI accelerator, ~11 TOPS INT8. Best for sustained low-power INT8 inference. Requires Level Zero NPU driver + OpenVINO NPU compiler.

- **GPU (RX 580)**: Discrete Polaris GPU, 8 GB VRAM. No ROCm support. Use via Vulkan (llama.cpp only).

## File structure

```
ai-benchmarks/
  01-system-info.py       Hardware detection, drivers, instruction sets
  02-pytorch-cpu.py       Matmul, thread scaling, transformer sim, mem bw
  03-openvino-all.py      CPU vs GPU vs NPU (matmul + CNN)
  04-llama-cpp-check.sh   Vulkan device listing for LLM inference
  run-all.sh              Run all benchmarks sequentially
  README.md               This file
```
