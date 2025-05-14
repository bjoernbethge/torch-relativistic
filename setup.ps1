# Setup-script for torch_relativistic with CUDA (Windows)
# Runs uv sync and installs the appropriate CUDA-Wheel for torch

# Check for uv
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install uv first."
    exit
}

# 2. Sync dependencies
uv sync

# 3. Install CUDA-compatible PyTorch (here: CUDA 12.6, Torch 2.6.0)
Write-Host "Install torch==2.6.0+cu126 ..."
uv pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

Write-Host "Done." 