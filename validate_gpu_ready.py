# save as: validate_gpu_ready.py
import torch
import sys
from pathlib import Path

print(" GPU Training Final Check\n")

# Check GPU
gpu_available = torch.cuda.is_available()
print(f" GPU Available: {gpu_available}")
if gpu_available:
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check key files exist
checks = {
    "Model Architecture": Path("src/models/timeseries_transformer.py").exists(),
    "Training Orchestrator": Path("src/training/trainer.py").exists(),
    "Data Collector": Path("src/data/collectors/yahoo_finance.py").exists(),
    "Loss Functions": Path("src/models/losses/composite_loss.py").exists(),
    "GPU Training Script": Path("scripts/training/train_single_gpu.py").exists(),
    "Backtest Engine": Path("src/backtesting/backtest_engine.py").exists(),
    "FastAPI Server": Path("src/api/main.py").exists(),
}

all_ready = all(checks.values())
for component, exists in checks.items():
    print(f"{'Y' if exists else 'X'} {component}")

print(f"\n{' READY FOR GPU TRAINING!' if all_ready else ' Missing components'}")