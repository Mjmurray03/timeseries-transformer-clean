#!/bin/bash
# VMware Setup Script
echo "Setting up TimeSeries Transformer on VMware"
echo "=========================================="

# Create virtual environment
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo "Setup complete!"
