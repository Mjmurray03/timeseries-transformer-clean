# Installation Guide

## Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) CUDA 11.8+ for GPU training

## Quick Install
```bash
bash install.sh
```

## Manual Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common Issues

### ImportError: No module named 'torch'
**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate
```

### CUDA not available
**Solution:** Install PyTorch with CUDA support
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Port 8000 already in use
**Solution:** Use a different port
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

## Verification
Run the test suite:
```bash
pytest tests/
```