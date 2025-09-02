# Save as verify_gpu.py on VMware
import torch
import sys
from pathlib import Path

print("=" * 60)
print("GPU ENVIRONMENT VERIFICATION")
print("=" * 60)

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU computation successful: {z.shape}")
    
    # Check memory after allocation
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
else:
    print("ERROR: CUDA not available!")
    print("Troubleshooting:")
    print("1. Check NVIDIA drivers: nvidia-smi")
    print("2. Reinstall PyTorch with CUDA support")
    sys.exit(1)

# Check project structure
print("\n" + "=" * 60)
print("PROJECT STRUCTURE CHECK")
print("=" * 60)

required_dirs = ["src", "data", "models", "scripts", "configs"]
for dir_name in required_dirs:
    exists = Path(dir_name).exists()
    print(f"{dir_name:10} : {'Found' if exists else 'MISSING'}")

# Check data files
data_files = list(Path("data/raw").glob("*.parquet"))
print(f"\nData files: {len(data_files)}/8")

print("\n" + "=" * 60)
if cuda_available and len(data_files) == 8:
    print("READY FOR GPU TRAINING!")
else:
    print("Issues found - fix before training")