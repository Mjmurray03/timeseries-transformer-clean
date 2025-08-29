#!/usr/bin/env python3
"""
Inference Script for Time-Series Transformer

Simple script to load trained model and make predictions on new data.
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import TimeSeriesTransformer
from data_loader import load_stock_data, FeatureEngineer, create_sequences
from utils import load_model_checkpoint, plot_predictions, calculate_metrics


def load_and_prepare_data(ticker: str, data_path: str, seq_length: int) -> tuple:
    """Load and prepare data for inference."""
    # Load raw data
    df = load_stock_data(data_path, ticker)
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df)
    df_clean = df_features.dropna()
    
    # Get last sequence for prediction
    feature_cols = feature_engineer.feature_names
    feature_data = df_clean[feature_cols].values.astype(np.float32)
    
    if len(feature_data) < seq_length:
        raise ValueError(f"Not enough data points. Need {seq_length}, got {len(feature_data)}")
    
    # Take the last sequence
    last_sequence = feature_data[-seq_length:]
    
    return last_sequence, feature_cols, df_clean


def make_prediction(model: torch.nn.Module, sequence: np.ndarray, device: str) -> np.ndarray:
    """Make prediction using the trained model."""
    model.eval()
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, seq_len, features)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy().squeeze()  # Remove batch dimension
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained transformer')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='Path to trained model')
    parser.add_argument('--data-path', type=str, default='data/raw', help='Path to data directory')
    parser.add_argument('--seq-length', type=int, default=60, help='Input sequence length')
    parser.add_argument('--show-plot', action='store_true', help='Show prediction plot')
    parser.add_argument('--num-features', type=int, default=8, help='Number of input features')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--forecast-horizon', type=int, default=3, help='Prediction horizon')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*50}")
    print(f"INFERENCE FOR {args.ticker}")
    print(f"{'='*50}")
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    try:
        last_sequence, feature_names, df_clean = load_and_prepare_data(
            args.ticker, args.data_path, args.seq_length
        )
        print(f"   Loaded data with {len(feature_names)} features")
        print(f"   Using last {args.seq_length} time steps for prediction")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create model with correct architecture
    print(f"\n2. Loading trained model...")
    model = TimeSeriesTransformer(
        input_dim=len(feature_names),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
        output_dim=args.forecast_horizon
    ).to(device)
    
    # Load model weights
    try:
        if Path(args.model).exists():
            load_model_checkpoint(model, args.model, device)
            print(f"   Model loaded from: {args.model}")
        else:
            print(f"Error: Model file not found: {args.model}")
            print("   Train a model first using: python scripts/train.py")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Make prediction
    print(f"\n3. Making predictions...")
    try:
        predictions = make_prediction(model, last_sequence, device)
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Predicted returns for next {args.forecast_horizon} days: {predictions}")
        
        # Convert to percentage changes if these are returns
        pred_percentages = [f"{p*100:.2f}%" for p in predictions]
        print(f"   Predicted percentage changes: {pred_percentages}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)
    
    # Optional: Show recent performance if we have test data
    print(f"\n4. Recent model performance (if available)...")
    try:
        # Create some test sequences for evaluation
        feature_data = df_clean[feature_names].values.astype(np.float32)
        
        if len(feature_data) > args.seq_length + args.forecast_horizon + 50:  # Need enough data
            # Use data from 50 steps back to create test sequences
            test_data = feature_data[-(50 + args.seq_length + args.forecast_horizon):-args.forecast_horizon]
            test_sequences, test_targets = create_sequences(
                test_data, args.seq_length, args.forecast_horizon, target_col=0
            )
            
            if len(test_sequences) > 0:
                # Make predictions on test data
                model.eval()
                test_predictions = []
                
                with torch.no_grad():
                    for seq in test_sequences:
                        input_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                        pred = model(input_tensor).cpu().numpy().squeeze()
                        test_predictions.append(pred[0])  # First prediction only
                
                test_predictions = np.array(test_predictions)
                test_actual = test_targets[:, 0]  # First target only
                
                # Calculate metrics
                metrics = calculate_metrics(test_actual, test_predictions)
                print(f"   Recent performance metrics:")
                for metric, value in metrics.items():
                    print(f"     {metric}: {value:.4f}")
                
                # Plot if requested
                if args.show_plot:
                    plot_predictions(
                        test_actual, 
                        test_predictions,
                        title=f"{args.ticker} - Model Performance",
                        save_path=f"results/predictions_{args.ticker}.png"
                    )
            else:
                print("   Not enough recent data for performance evaluation")
        else:
            print("   Not enough historical data for performance evaluation")
            
    except Exception as e:
        print(f"   Could not evaluate recent performance: {e}")
    
    print(f"\n[SUCCESS] Prediction complete!")
    print(f"   Next {args.forecast_horizon}-day return forecast: {predictions}")


if __name__ == "__main__":
    main()