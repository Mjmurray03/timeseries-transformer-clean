#!/usr/bin/env python3
"""
SIMPLE GPU Training Script - No BS Edition
Just trains the model. That's it.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer
from src.data.datasets.stock_dataset import StockSequenceDataset
from src.data.processors.feature_engineering import FeatureEngineer

def load_data():
    """Load and prepare data - simple version"""
    print("Loading data...")
    
    # Load all parquet files
    data_files = list(Path('data/raw').glob('*.parquet'))
    if not data_files:
        raise FileNotFoundError("No data files found in data/raw/")
    
    all_data = []
    for file in data_files:
        df = pd.read_parquet(file)
        # Add ticker from filename
        ticker = file.stem.split('_')[0]
        df['ticker'] = ticker
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows of data")
    
    # Engineer features
    print("Engineering features...")
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(combined_df)
    
    # Remove any NaN values that might cause issues
    features_df = features_df.dropna()
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in features_df.columns 
                   if col not in ['ticker', 'Date', 'date']]
    
    # Convert to numpy for simplicity
    data_array = features_df[feature_cols].values.astype(np.float32)
    
    # Create sequences
    print("Creating sequences...")
    seq_len = 60
    horizon = 5
    sequences = []
    targets = []
    
    for i in range(seq_len, len(data_array) - horizon):
        sequences.append(data_array[i-seq_len:i])
        # Just predict close price (column 3 typically)
        targets.append(data_array[i:i+horizon, 3])  
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    return sequences, targets, features_df.shape[1] - 1  # -1 for ticker column

def main():
    parser = argparse.ArgumentParser(description='Simple GPU Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    sequences, targets, n_features = load_data()
    
    # Create dataset
    dataset = StockSequenceDataset(sequences, targets)
    
    # Split data (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {train_size} samples, Val: {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model (SIMPLE!)
    model = TimeSeriesTransformer(
        input_dim=n_features,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        forecast_horizon=5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simple optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop (SIMPLE!)
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Handle both dict and tuple returns from dataset
            if isinstance(batch, dict):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle model output format
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get predictions, ignore attention
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    inputs = batch['inputs'].to(device)
                    targets = batch['targets'].to(device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Print epoch summary
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print("-"*50)
        
        # Simple checkpoint saving
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'models/checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = 'models/final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    main()