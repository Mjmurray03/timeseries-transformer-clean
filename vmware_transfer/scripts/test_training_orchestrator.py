#!/usr/bin/env python3
"""Test script for TrainingOrchestrator implementation."""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.training_config import TrainingConfig, create_quick_test_config
from training.trainer import TrainingOrchestrator
from data.datasets.stock_dataset import StockSequenceDataset, create_data_loaders


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Use last timestep for prediction
        x = x[:, -1, :]
        return self.output_projection(x)


def create_dummy_data(num_samples=1000, seq_len=60, num_features=7, horizon=5):
    """Create dummy stock data for testing."""
    # Generate random sequences
    sequences = np.random.randn(num_samples, seq_len, num_features).astype(np.float32)
    
    # Generate targets (simple linear combination for testing)
    targets = np.random.randn(num_samples, horizon).astype(np.float32)
    
    return sequences, targets


def test_training_orchestrator():
    """Test the TrainingOrchestrator implementation."""
    print("Testing TrainingOrchestrator...")
    
    # Create dummy data
    print("Creating dummy data...")
    sequences, targets = create_dummy_data(num_samples=200, seq_len=60, num_features=7, horizon=5)
    
    # Split data
    train_size = int(0.7 * len(sequences))
    val_size = int(0.15 * len(sequences))
    
    train_seq, train_targets = sequences[:train_size], targets[:train_size]
    val_seq, val_targets = sequences[train_size:train_size+val_size], targets[train_size:train_size+val_size]
    test_seq, test_targets = sequences[train_size+val_size:], targets[train_size+val_size:]
    
    # Create datasets
    train_dataset = StockSequenceDataset(train_seq, train_targets)
    val_dataset = StockSequenceDataset(val_seq, val_targets)
    test_dataset = StockSequenceDataset(test_seq, test_targets)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=16, num_workers=0  # Use 0 workers for testing
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    # Create model
    print("Creating model...")
    model = SimpleTransformer(input_dim=7, hidden_dim=64, output_dim=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training config
    config = create_quick_test_config()
    config.num_epochs = 3  # Very short for testing
    config.log_every = 5
    config.use_amp = False  # Disable AMP for CPU testing
    config.device = "cpu"  # Force CPU for testing
    config.experiment_name = "test_training_orchestrator"
    
    print(f"Training config: {config.num_epochs} epochs, batch size {config.batch_size}")
    
    # Create training orchestrator
    print("Creating TrainingOrchestrator...")
    trainer = TrainingOrchestrator(model, config)
    
    # Run training
    print("Starting training...")
    try:
        results = trainer.train(train_loader, val_loader, test_loader)
        
        print("Training completed successfully!")
        print(f"Results: {results}")
        
        # Test checkpoint saving/loading
        print("Testing checkpoint functionality...")
        checkpoint_path = "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, 0, {'test_metric': 0.5})
        
        # Create new trainer and load checkpoint
        new_trainer = TrainingOrchestrator(SimpleTransformer(), config)
        checkpoint = new_trainer.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training_orchestrator()
    if success:
        print("\n✅ TrainingOrchestrator test passed!")
        sys.exit(0)
    else:
        print("\n❌ TrainingOrchestrator test failed!")
        sys.exit(1)