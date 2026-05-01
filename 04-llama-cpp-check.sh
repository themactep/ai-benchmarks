#!/bin/bash
# List all Vulkan compute devices available for llama.cpp
# Usage: ./04-llama-cpp-check.sh

set -e

LLAMA_BUILD="$HOME/llama.cpp/build"
LLAMA_CLI="$LLAMA_BUILD/bin/llama-cli"

if [ ! -f "$LLAMA_CLI" ]; then
    echo "llama.cpp not found at $LLAMA_CLI"
    echo ""
    echo "To build llama.cpp with Vulkan support:"
    echo "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp"
    echo "  cmake -B $LLAMA_BUILD -DGGML_VULKAN=ON -S ~/llama.cpp"
    echo "  cmake --build $LLAMA_BUILD --config Release -j\$(nproc)"
    exit 1
fi

echo "=== llama.cpp Vulkan Devices ==="
"$LLAMA_CLI" --list-devices 2>&1

echo ""
echo "=== Usage Examples ==="
echo ""
echo "# Run a model on Intel Arc GPU (Vulkan0):"
echo "$LLAMA_CLI -m /path/to/model.gguf -ngl 99 --device Vulkan0 -p \"Your prompt\""
echo ""
echo "# Run a model on AMD RX 580 (Vulkan1):"
echo "$LLAMA_CLI -m /path/to/model.gguf -ngl 99 --device Vulkan1 -p \"Your prompt\""
echo ""
echo "# -ngl 99 = offload all layers to GPU"
echo "# Omit --device flag to auto-select the first Vulkan device"
