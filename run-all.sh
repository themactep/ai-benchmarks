#!/bin/bash
# Run all AI benchmarks sequentially
# Usage: ./run-all.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════════╗"
echo "║         AI BENCHMARKS — FULL SUITE                  ║"
echo "╚══════════════════════════════════════════════════════╝"

echo ""
echo "1/4 System Info"
python3 "$SCRIPT_DIR/01-system-info.py"

echo ""
echo "2/4 PyTorch CPU"
python3 "$SCRIPT_DIR/02-pytorch-cpu.py"

echo ""
echo "3/4 OpenVINO CPU/GPU/NPU"
python3 "$SCRIPT_DIR/03-openvino-all.py"

echo ""
echo "4/4 llama.cpp Vulkan"
bash "$SCRIPT_DIR/04-llama-cpp-check.sh"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         ALL BENCHMARKS COMPLETE                     ║"
echo "╚══════════════════════════════════════════════════════╝"
