#!/usr/bin/env python3
"""
Clean Training Script for Time-Series Transformer

Simplified training loop with proper logging and model saving.
Based on the working train_ultra_simple.py but cleaned for academic presentation.
"""

import torch
import torch.nn as nn
import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import TimeSeriesTransformer
from data_loader import prepare_data
from utils import (
    set_seed, 
    plot_training_curves, 
    calculate_metrics, 
    save_model_checkpoint,
    save_model_info,
    save_metrics,
    EarlyStopping,
    format_time,
    print_model_summary
)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 20 batches
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx:3d}/{len(train_loader):3d} | Loss: {loss.item():.6f}")
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Time-Series Transformer')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--seq-length', type=int, default=60, help='Input sequence length')
    parser.add_argument('--forecast-horizon', type=int, default=3, help='Prediction horizon')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-path', type=str, default='data/raw', help='Path to data directory')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING TIME-SERIES TRANSFORMER FOR {args.ticker}")
    print(f"{'='*60}")
    
    # Load data
    print("\n1. Loading and preparing data...")
    try:
        train_loader, test_loader, num_features = prepare_data(
            ticker=args.ticker,
            data_path=args.data_path,
            seq_length=args.seq_length,
            forecast_horizon=args.forecast_horizon,
            test_size=0.2
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure data files exist in {args.data_path}")
        sys.exit(1)
    
    # Create model
    print(f"\n2. Creating model...")
    model = TimeSeriesTransformer(
        input_dim=num_features,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
        output_dim=args.forecast_horizon
    ).to(device)
    
    # Print model summary
    print_model_summary(model, (args.batch_size, args.seq_length, num_features))
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=7, min_delta=1e-6)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n3. Starting training...")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Forecast horizon: {args.forecast_horizon}")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss = validate_epoch(model, test_loader, criterion, device)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time:       {format_time(epoch_time)}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/best_model.pt')
            print(f"  [BEST] New best model saved!")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    
    # Generate final predictions for evaluation
    print("\n4. Evaluating final model...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    import numpy as np
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics (using first time step of predictions)
    metrics = calculate_metrics(targets[:, 0], predictions[:, 0])
    print(f"\nFinal Metrics (1-day forecast):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    print("\n5. Saving results...")
    
    # Save model info and metrics
    save_model_info(model, metrics, 'models/model_info.json')
    save_metrics(metrics, 'results/metrics.txt')
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, 'results/training_curves.png')
    
    # Save final model checkpoint
    save_model_checkpoint(
        model, optimizer, len(train_losses), 
        train_losses[-1], val_losses[-1],
        'models/final_checkpoint.pt'
    )
    
    print(f"\n[SUCCESS] Training complete! Results saved to:")
    print(f"   Model: models/best_model.pt")
    print(f"   Metrics: results/metrics.txt")
    print(f"   Plots: results/training_curves.png")
    
    return best_val_loss


if __name__ == "__main__":
    main()