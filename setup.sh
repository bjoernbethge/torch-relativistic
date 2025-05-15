#!/bin/bash
# Setup-script for torch_relativistic with CUDA (Linux)
# Runs uv sync and installs the appropriate CUDA-Wheel for torch

set -e

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Install uv first."
    exit 1
fi

# 2. Sync dependencies
uv sync 

# 3. Install CUDA-compatible PyTorch (here: CUDA 12.6, Torch 2.6.0)
echo "Install torch==2.6.0+cu126 ..."
uv pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

echo "Done." 