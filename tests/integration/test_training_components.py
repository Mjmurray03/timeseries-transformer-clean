# tests/integration/test_training_components.py
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.trainer import TrainingOrchestrator
from src.models.losses.composite_loss import CompositeLoss
from src.data.datasets.stock_dataset import StockSequenceDataset, create_data_loaders, split_sequences
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.config.training_config import TrainingConfig
import pandas as pd
import numpy as np

def test_training_components():
    """Test training pipeline components."""
    
    print("[*] Testing Training Pipeline Components...")
    
    # Test 1: CompositeLoss
    print("\n[1] Testing CompositeLoss...")
    loss_fn = CompositeLoss(
        price_weight=0.4,
        direction_weight=0.3,
        volatility_weight=0.2,
        quantile_weight=0.1
    )
    
    batch_size = 16
    forecast_horizon = 5
    
    # Create dummy predictions and targets matching CompositeLoss expectations
    predictions = {
        'price': torch.randn(batch_size, forecast_horizon),
        'volatility': torch.randn(batch_size, forecast_horizon),
        'quantiles': torch.randn(batch_size, forecast_horizon, 5)  # 5 quantiles
    }
    
    targets = {
        'price': torch.randn(batch_size, forecast_horizon),
        'volatility': torch.randn(batch_size, forecast_horizon)
    }
    
    loss, loss_components = loss_fn(predictions, targets)
    assert loss is not None, "[X] Loss calculation failed"
    assert loss.item() > 0, "[X] Loss should be positive"
    assert 'price_loss' in loss_components, "[X] Missing price_loss component"
    assert 'direction_loss' in loss_components, "[X] Missing direction_loss component"
    assert 'volatility_loss' in loss_components, "[X] Missing volatility_loss component"
    assert 'quantile_loss' in loss_components, "[X] Missing quantile_loss component"
    print(f"[OK] CompositeLoss working: {loss.item():.4f}")
    print(f"     Components: price={loss_components['price_loss']:.4f}, direction={loss_components['direction_loss']:.4f}")
    
    # Test 2: DataLoader Creation using existing StockSequenceDataset
    print("\n[2] Testing DataLoader Creation...")
    
    # Create realistic dummy time series data
    n_samples = 1000
    sequence_length = 60
    num_features = 10
    
    # Generate sequences with proper temporal structure
    dates = pd.date_range('2024-01-01', periods=n_samples + sequence_length, freq='D')
    
    # Generate realistic price movements
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    
    # Create features (OHLCV + technical indicators)
    ohlcv_data = np.column_stack([
        prices * (1 + np.random.normal(0, 0.001, len(prices))),  # Open
        prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),  # High
        prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),  # Low
        prices,  # Close
        np.random.randint(1000000, 10000000, len(prices)),  # Volume
    ])
    
    # Add technical indicators (RSI, MACD, etc.)
    technical_indicators = np.random.randn(len(prices), 5)
    all_features = np.column_stack([ohlcv_data, technical_indicators])
    
    # Create sequences and targets
    sequences = []
    targets = []
    
    for i in range(n_samples):
        # Check if we have enough data for both sequence and target
        if i + sequence_length + 5 <= len(prices):
            # Input sequence
            seq = all_features[i:i + sequence_length]
            sequences.append(seq)
            
            # Target (next 5 days closing prices)
            target_prices = prices[i + sequence_length:i + sequence_length + 5]
            targets.append(target_prices)
    
    # Update n_samples to actual number of valid samples
    n_samples = len(sequences)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split data
    splits = split_sequences(sequences, targets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets) = splits
    
    # Create datasets
    train_dataset = StockSequenceDataset(train_seq, train_targets)
    val_dataset = StockSequenceDataset(val_seq, val_targets)
    test_dataset = StockSequenceDataset(test_seq, test_targets)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=32, num_workers=0  # Use 0 workers on Windows for simplicity
    )
    
    assert train_loader is not None, "[X] Train loader creation failed"
    assert val_loader is not None, "[X] Val loader creation failed"
    assert test_loader is not None, "[X] Test loader creation failed"
    
    # Test one batch
    for batch in train_loader:
        assert 'inputs' in batch, "[X] Missing inputs in batch"
        assert 'targets' in batch, "[X] Missing targets in batch"
        assert batch['inputs'].shape[1] == 60, "[X] Wrong sequence length"
        assert batch['inputs'].shape[2] == 10, "[X] Wrong number of features"
        print(f"[OK] DataLoader working: batch shape {batch['inputs'].shape}")
        break
    
    # Test 3: TrainingOrchestrator Initialization
    print("\n[3] Testing TrainingOrchestrator...")
    
    # Create model
    model = TimeSeriesTransformer(
        input_dim=10,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_length=60,
        output_dim=5,  # Predict 5-day horizon
        forecast_horizon=5
    )
    
    # Create minimal training config using the dataclass structure
    from src.config.training_config import OptimizerConfig, SchedulerConfig, LossConfig
    
    optimizer_config = OptimizerConfig(
        name='adamw',
        learning_rate=0.001,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False
    )
    
    scheduler_config = SchedulerConfig(
        name='cosine',
        max_steps=100,
        min_lr=1e-6,
        patience=5,
        factor=0.5
    )
    
    loss_config = LossConfig()
    
    config = TrainingConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=10,
        batch_size=32,
        val_every=1,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-4,
        checkpoint_dir='./test_checkpoints',
        save_best_only=True,
        use_amp=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        gradient_clip=1.0,
        log_every=10,
        deterministic=True,
        seed=42,
        experiment_name='test_training_components',
        project_name='timeseries-transformer-test',
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        loss=loss_config
    )
    
    orchestrator = TrainingOrchestrator(
        model=model,
        config=config
    )
    
    assert orchestrator is not None, "[X] Orchestrator initialization failed"
    assert orchestrator.model is not None, "[X] Model not attached"
    assert orchestrator.optimizer is not None, "[X] Optimizer not created"
    assert orchestrator.scheduler is not None, "[X] Scheduler not created"
    assert orchestrator.criterion is not None, "[X] Criterion not created"
    print(f"[OK] TrainingOrchestrator initialized on {orchestrator.device}")
    
    # Test forward pass through the model
    print("\n[4] Testing Model Forward Pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        inputs = sample_batch['inputs']
        
        # Test forward pass
        outputs = model(inputs)
        
        assert outputs is not None, "[X] Model forward pass failed"
        assert outputs.shape[0] == inputs.shape[0], "[X] Batch size mismatch"
        assert outputs.shape[1] == 5, "[X] Output dimension mismatch"
        print(f"[OK] Model forward pass: input {inputs.shape} -> output {outputs.shape}")
    
    print("\n[SUCCESS] Training components test PASSED!")
    return True

if __name__ == "__main__":
    test_training_components()