# tests/integration/test_end_to_end.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import yaml

def test_end_to_end_pipeline():
    """Test complete pipeline from data to training."""
    
    print("[*] Testing End-to-End Pipeline...")
    print("="*50)
    
    # Step 1: Data Collection with Doppler-managed secrets
    print("\n[DATA] Step 1: Collecting Data...")
    from src.data.collectors.yahoo_finance import YahooFinanceCollector
    from src.config.data_config import DataConfig
    
    # Load configuration with Doppler integration
    config_path = Path(__file__).parent.parent.parent / "configs" / "data_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = DataConfig(config_dict)
    
    collector = YahooFinanceCollector(config)
    
    # Use async collection as per actual implementation
    async def collect_data():
        return await collector.collect_ticker(
            ticker='AAPL',
            start_date=(datetime.now() - timedelta(days=400)).date(),  # Need enough for validation
            end_date=datetime.now().date()
        )
    
    data = asyncio.run(collect_data())
    assert data is not None, "[X] Data collection failed"
    print(f"[OK] Collected {len(data)} days of data")
    
    # Step 2: Data Validation
    print("\n[VALIDATION] Step 2: Validating Data...")
    from src.data.validators import DataValidator
    
    validator = DataValidator(config)
    validation_result = validator.validate(data, 'AAPL')
    assert validation_result.is_valid, f"[X] Data validation failed: {validation_result.issues}"
    print("[OK] Data validation passed")
    
    # Step 3: Feature Engineering  
    print("\n[FEATURES] Step 3: Engineering Features...")
    from src.data.processors.feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features(data)
    assert len(engineered_data.columns) > len(data.columns), "[X] No features added"
    print(f"[OK] Engineered {len(engineered_data.columns)} features (added {len(engineered_data.columns) - len(data.columns)})")
    
    # Step 4: Create Dataset and DataLoader using REAL existing components
    print("\n[DATASET] Step 4: Creating Dataset...")
    from src.data.datasets.stock_dataset import StockSequenceDataset, create_data_loaders, split_sequences
    
    # Build sequences manually since TimeSeriesDataLoader doesn't exist
    sequence_length = 60
    forecast_horizon = 5
    
    # Convert data to numpy for sequence building with proper data types
    numeric_data = engineered_data.select_dtypes(include=[np.number])
    
    # Simple normalization to prevent numerical instability
    # Remove NaN and inf values first
    numeric_data = numeric_data.fillna(0)  # Fill NaN with 0
    numeric_data = numeric_data.replace([np.inf, -np.inf], 0)  # Replace inf with 0
    
    # Standard normalization (z-score)
    feature_data = (numeric_data - numeric_data.mean()) / (numeric_data.std() + 1e-8)
    feature_data = feature_data.values.astype(np.float32)
    
    sequences = []
    targets = []
    
    for i in range(len(feature_data) - sequence_length - forecast_horizon + 1):
        # Input sequence
        seq = feature_data[i:i + sequence_length]
        sequences.append(seq)
        
        # Target (next forecast_horizon days closing prices)
        # Assume Close is the 4th column (index 3)
        target_prices = feature_data[i + sequence_length:i + sequence_length + forecast_horizon, 3]  # Close prices
        targets.append(target_prices)
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
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
        batch_size=16, num_workers=0  # 0 workers on Windows
    )
    
    print(f"[OK] Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Step 5: Initialize Model using REAL implementation
    print("\n[MODEL] Step 5: Initializing Model...")
    from src.models.timeseries_transformer import TimeSeriesTransformer
    
    # Use the actual number of numeric features
    actual_input_dim = feature_data.shape[1]
    
    model = TimeSeriesTransformer(
        input_dim=actual_input_dim,  # Use actual numeric feature count
        hidden_dim=64,  # Small for testing
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        max_seq_length=60,
        output_dim=forecast_horizon,  # Predict 5-day horizon
        forecast_horizon=forecast_horizon
    )
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model initialized with {params:,} parameters")
    
    # Step 6: Setup Training using REAL TrainingOrchestrator
    print("\n[TRAINING] Step 6: Setting Up Training...")
    from src.training.trainer import TrainingOrchestrator
    from src.config.training_config import TrainingConfig, OptimizerConfig, SchedulerConfig, LossConfig
    
    # Create proper training config
    optimizer_config = OptimizerConfig(
        name='adamw',
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    scheduler_config = SchedulerConfig(
        name='cosine',
        max_steps=100,
        min_lr=1e-6
    )
    
    loss_config = LossConfig()
    
    training_config = TrainingConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=2,  # Just 2 epochs for testing
        batch_size=16,
        val_every=1,
        early_stopping_patience=5,
        checkpoint_dir='./test_checkpoints',
        save_best_only=True,
        use_amp=False,  # Disable for testing stability
        gradient_accumulation_steps=1,
        gradient_clip=1.0,
        log_every=10,
        deterministic=True,
        seed=42,
        experiment_name='end_to_end_test',
        project_name='timeseries-transformer-test',
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        loss=loss_config
    )
    
    orchestrator = TrainingOrchestrator(
        model=model,
        config=training_config
    )
    print(f"[OK] Training orchestrator ready on {orchestrator.device}")
    
    # Step 7: Run Training Loop
    print("\n[TRAIN] Step 7: Running Training Loop...")
    print("-"*30)
    
    # Ensure checkpoint directory exists
    checkpoint_path = Path('./test_checkpoints')
    checkpoint_path.mkdir(exist_ok=True)
    
    model.train()
    train_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Only 5 batches for testing
            break
            
        # Move to device
        inputs = batch['inputs'].to(orchestrator.device)
        targets = batch['targets'].to(orchestrator.device)
        
        # Forward pass
        orchestrator.optimizer.zero_grad()
        predictions = model(inputs)  # Get predictions
        
        # Calculate loss using simple MSE (matching TrainingOrchestrator's current criterion)
        loss = orchestrator.criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        orchestrator.optimizer.step()
        
        train_losses.append(loss.item())
        print(f"   Batch {batch_idx+1}/5: Loss = {loss.item():.4f}")
    
    avg_loss = np.mean(train_losses)
    print(f"[OK] Training loop working: Avg Loss = {avg_loss:.4f}")
    
    # Step 8: Validation
    print("\n[VALIDATION] Step 8: Running Validation...")
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 3:  # Only 3 batches for testing
                break
                
            inputs = batch['inputs'].to(orchestrator.device)
            targets = batch['targets'].to(orchestrator.device)
            
            predictions = model(inputs)
            loss = orchestrator.criterion(predictions, targets)
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    print(f"[OK] Validation working: Avg Loss = {avg_val_loss:.4f}")
    
    # Step 9: Save Checkpoint using TrainingOrchestrator method
    print("\n[CHECKPOINT] Step 9: Saving Checkpoint...")
    checkpoint_file = checkpoint_path / 'test_model.pt'
    
    orchestrator.save_checkpoint(
        str(checkpoint_file),
        epoch=1,
        metrics={'train_loss': avg_loss, 'val_loss': avg_val_loss}
    )
    
    assert checkpoint_file.exists(), "[X] Checkpoint not saved"
    print(f"[OK] Checkpoint saved to {checkpoint_file}")
    
    # Step 10: Load Checkpoint using TrainingOrchestrator method
    print("\n[LOAD] Step 10: Loading Checkpoint...")
    checkpoint_data = orchestrator.load_checkpoint(str(checkpoint_file))
    
    assert checkpoint_data['epoch'] == 1, "[X] Wrong epoch loaded"
    assert 'train_loss' in checkpoint_data['metrics'], "[X] Missing training metrics"
    print(f"[OK] Checkpoint loaded: Epoch {checkpoint_data['epoch']}, Metrics: {checkpoint_data['metrics']}")
    
    # Step 11: Test inference on new data
    print("\n[INFERENCE] Step 11: Testing Inference...")
    model.eval()
    with torch.no_grad():
        # Get one test batch
        test_batch = next(iter(test_loader))
        test_inputs = test_batch['inputs'].to(orchestrator.device)
        test_targets = test_batch['targets'].to(orchestrator.device)
        
        # Make predictions
        test_predictions = model(test_inputs)
        
        # Calculate test metrics
        test_loss = orchestrator.criterion(test_predictions, test_targets)
        
        print(f"[OK] Inference working: Test Loss = {test_loss.item():.4f}")
        print(f"     Input shape: {test_inputs.shape}, Output shape: {test_predictions.shape}")
    
    # Cleanup
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    if checkpoint_path.exists() and not any(checkpoint_path.iterdir()):
        checkpoint_path.rmdir()
    
    print("\n"+"="*50)
    print("[SUCCESS] END-TO-END PIPELINE TEST PASSED!")
    print("="*50)
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"- Data: {len(data)} days collected, {actual_input_dim} numeric features")
    print(f"- Model: {params:,} parameters")
    print(f"- Training: {avg_loss:.4f} avg loss over {len(train_losses)} batches")
    print(f"- Validation: {avg_val_loss:.4f} avg loss")
    print(f"- Test: {test_loss.item():.4f} inference loss")
    
    return True

if __name__ == "__main__":
    test_end_to_end_pipeline()