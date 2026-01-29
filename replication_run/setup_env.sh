#!/bin/bash
# Setup script for emotion detection pipeline using uv

set -e  # Exit on error

echo "========================================"
echo "Emotion Detection Pipeline - UV Setup"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
uv venv emotion_env --python 3.11

echo "✓ Virtual environment created at emotion_env"

# Activate environment (for this script)
source emotion_env/bin/activate

# Install PyTorch with CUDA support first
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "Installing other dependencies..."
uv pip install transformers numpy pandas scikit-learn matplotlib

# Verify CUDA is available
echo ""
echo "Verifying CUDA availability..."
python -c "
import torch
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')
if cuda_available:
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA is not available! Training will run on CPU.')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source emotion_env/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python project_part_b.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
