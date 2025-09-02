#!/usr/bin/env python3
"""
Phase 5: Model Validation Script
Purpose: Validate trained models with comprehensive metrics and visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


class ModelValidator:
    """Validate trained models on test data"""
    
    def __init__(self, model_path: str, data_dir: str = "data/raw", scalers_dir: str = "scalers"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.scalers_dir = Path(scalers_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        print(f"Loading model from {self.model_path}...")
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        self.model_info = self._extract_model_info()
        
        # Initialize model
        self.model = self._load_model()
        
        # Load scalers
        self.scalers = self._load_scalers()
        
    def _extract_model_info(self) -> Dict:
        """Extract information from checkpoint"""
        info = {
            'epoch': self.checkpoint.get('epoch', 'Unknown'),
            'val_loss': self.checkpoint.get('val_loss', 'Unknown'),
            'val_rmse': self.checkpoint.get('val_rmse', 'Unknown'),
            'config': self.checkpoint.get('config', {}),
            'timestamp': self.checkpoint.get('timestamp', 'Unknown')
        }
        
        # Determine if single or multi-stock
        if 'tickers' in self.checkpoint:
            info['type'] = 'multi_stock'
            info['tickers'] = self.checkpoint['tickers']
        elif 'ticker' in self.checkpoint:
            info['type'] = 'single_stock'
            info['tickers'] = [self.checkpoint['ticker']]
        else:
            # Try to infer from filename
            if 'multi' in str(self.model_path):
                info['type'] = 'multi_stock'
                info['tickers'] = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NFLX', 'NVDA', 'TSLA']
            else:
                info['type'] = 'single_stock'
                ticker = str(self.model_path).split('_')[1] if '_' in str(self.model_path) else 'AAPL'
                info['tickers'] = [ticker]
        
        return info
    
    def _load_model(self) -> nn.Module:
        """Load the model architecture and weights"""
        # Default dimensions if not in config
        input_dim = self.model_info['config'].get('input_dim', 10)
        hidden_dim = self.model_info['config'].get('hidden_dim', 256)
        num_layers = self.model_info['config'].get('num_layers', 6)
        num_heads = self.model_info['config'].get('num_heads', 8)
        output_dim = self.model_info['config'].get('horizon', 3)
        dropout = self.model_info['config'].get('dropout', 0.1)
        
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            dropout=dropout
        ).to(self.device)
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _load_scalers(self) -> Dict:
        """Load scaler parameters for each ticker"""
        scalers = {}
        for ticker in self.model_info['tickers']:
            scaler_path = self.scalers_dir / f"scaler_{ticker}.json"
            if scaler_path.exists():
                with open(scaler_path, 'r') as f:
                    scalers[ticker] = json.load(f)
            else:
                print(f"Warning: Scaler not found for {ticker}")
        return scalers
    
    def load_test_data(self, ticker: str, test_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare test data for a ticker"""
        # Load raw data
        data_file = self.data_dir / f"{ticker}.parquet"
        if not data_file.exists():
            data_file = self.data_dir / f"{ticker.lower()}.parquet"
        
        df = pd.read_parquet(data_file)
        df = df.sort_index()
        
        # Take last test_size rows for testing
        df_test = df.iloc[-test_size:]
        
        # Engineer features (same as training)
        features = self._engineer_features(df_test)
        
        # Normalize using saved scaler
        if ticker in self.scalers:
            scaler = self.scalers[ticker]
            mean = np.array(scaler['mean'])
            std = np.array(scaler['std'])
            normalized = (features - mean) / std
        else:
            # Fallback normalization
            normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Create sequences
        seq_len = self.model_info['config'].get('seq_len', 60)
        horizon = self.model_info['config'].get('horizon', 3)
        
        sequences = []
        targets = []
        
        for i in range(seq_len, len(normalized) - horizon):
            sequences.append(normalized[i-seq_len:i])
            targets.append(normalized[i:i+horizon, 3])  # Close price
        
        if len(sequences) == 0:
            return None, None
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features from raw data"""
        features = []
        
        features.append(df['Open'].values)
        features.append(df['High'].values)
        features.append(df['Low'].values)
        features.append(df['Close'].values)
        features.append(df['Volume'].values)
        
        # Technical indicators
        features.append(df['Close'].pct_change().fillna(0).values)
        features.append(df['Close'].rolling(5).mean().fillna(df['Close']).values)
        features.append(df['Close'].rolling(20).mean().fillna(df['Close']).values)
        
        returns = df['Close'].pct_change()
        features.append(returns.rolling(20).std().fillna(0).values)
        features.append(df['Volume'].rolling(5).mean().fillna(df['Volume']).values)
        
        return np.stack(features, axis=1)
    
    def validate_ticker(self, ticker: str) -> Dict:
        """Validate model on a single ticker"""
        print(f"\nValidating {ticker}...")
        
        # Load test data
        sequences, targets = self.load_test_data(ticker)
        
        if sequences is None:
            return {'ticker': ticker, 'error': 'Insufficient data'}
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences_tensor)
        
        # Calculate metrics
        mse = torch.mean((predictions - targets_tensor) ** 2).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - targets_tensor)).item()
        
        # Direction accuracy
        pred_direction = (predictions[:, 0] > 0).float()
        true_direction = (targets_tensor[:, 0] > 0).float()
        direction_acc = (pred_direction == true_direction).float().mean().item()
        
        # Convert back to prices for interpretability
        if ticker in self.scalers:
            scaler = self.scalers[ticker]
            close_mean = scaler['mean'][3]
            close_std = scaler['std'][3]
            
            # Denormalize
            predictions_price = predictions.cpu().numpy() * close_std + close_mean
            targets_price = targets * close_std + close_mean
            
            # Price RMSE
            price_rmse = np.sqrt(np.mean((predictions_price - targets_price) ** 2))
        else:
            price_rmse = None
        
        return {
            'ticker': ticker,
            'num_samples': len(sequences),
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            'price_rmse': price_rmse,
            'predictions': predictions.cpu().numpy(),
            'targets': targets
        }
    
    def validate_all(self) -> Dict:
        """Validate on all tickers"""
        results = {
            'model_path': str(self.model_path),
            'model_type': self.model_info['type'],
            'training_epoch': self.model_info['epoch'],
            'training_val_loss': self.model_info['val_loss'],
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'ticker_results': {}
        }
        
        for ticker in self.model_info['tickers']:
            ticker_results = self.validate_ticker(ticker)
            results['ticker_results'][ticker] = ticker_results
            
            # Print summary
            if 'error' not in ticker_results:
                print(f"  RMSE: {ticker_results['rmse']:.4f}")
                print(f"  MAE: {ticker_results['mae']:.4f}")
                print(f"  Direction Accuracy: {ticker_results['direction_accuracy']:.2%}")
                if ticker_results['price_rmse']:
                    print(f"  Price RMSE: ${ticker_results['price_rmse']:.2f}")
        
        # Calculate aggregate metrics
        valid_results = [r for r in results['ticker_results'].values() if 'error' not in r]
        
        if valid_results:
            results['aggregate'] = {
                'avg_rmse': np.mean([r['rmse'] for r in valid_results]),
                'avg_mae': np.mean([r['mae'] for r in valid_results]),
                'avg_direction_accuracy': np.mean([r['direction_accuracy'] for r in valid_results]),
                'total_samples': sum([r['num_samples'] for r in valid_results])
            }
            
            price_rmses = [r['price_rmse'] for r in valid_results if r['price_rmse']]
            if price_rmses:
                results['aggregate']['avg_price_rmse'] = np.mean(price_rmses)
        
        return results
    
    def plot_predictions(self, ticker: str, num_samples: int = 20):
        """Plot actual vs predicted values"""
        results = self.validate_ticker(ticker)
        
        if 'error' in results:
            print(f"Cannot plot {ticker}: {results['error']}")
            return
        
        predictions = results['predictions'][:num_samples]
        targets = results['targets'][:num_samples]
        
        plt.figure(figsize=(15, 5))
        
        # Plot each horizon step
        for h in range(predictions.shape[1]):
            plt.subplot(1, 3, h+1)
            plt.scatter(targets[:, h], predictions[:, h], alpha=0.5)
            
            # Perfect prediction line
            min_val = min(targets[:, h].min(), predictions[:, h].min())
            max_val = max(targets[:, h].max(), predictions[:, h].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
            
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{ticker} - Horizon {h+1}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'validation_{ticker}_{self.model_info["type"]}.png')
        plt.show()
        print(f"Saved plot: validation_{ticker}_{self.model_info['type']}.png")
    
    def save_report(self, results: Dict, output_dir: str = "validation_results"):
        """Save validation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"validation_{self.model_info['type']}_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_arrays(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist() if isinstance(obj, np.ndarray) else float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_arrays(item) for item in obj]
                return obj
            
            json.dump(convert_arrays(results), f, indent=2)
        
        print(f"\nValidation report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Model Type: {self.model_info['type']}")
        print(f"Tickers: {', '.join(self.model_info['tickers'])}")
        print(f"Training Epochs: {self.model_info['epoch']}")
        print(f"Training Val Loss: {self.model_info['val_loss']}")
        
        if 'aggregate' in results:
            print(f"\nAggregate Metrics:")
            print(f"  Average RMSE: {results['aggregate']['avg_rmse']:.4f}")
            print(f"  Average MAE: {results['aggregate']['avg_mae']:.4f}")
            print(f"  Average Direction Accuracy: {results['aggregate']['avg_direction_accuracy']:.2%}")
            if 'avg_price_rmse' in results['aggregate']:
                print(f"  Average Price RMSE: ${results['aggregate']['avg_price_rmse']:.2f}")
            print(f"  Total Test Samples: {results['aggregate']['total_samples']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate trained models')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Data directory')
    parser.add_argument('--scalers-dir', type=str, default='scalers',
                        help='Scalers directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate prediction plots')
    parser.add_argument('--ticker', type=str, default=None,
                        help='Specific ticker to plot')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ModelValidator(
        model_path=args.model,
        data_dir=args.data_dir,
        scalers_dir=args.scalers_dir
    )
    
    # Run validation
    results = validator.validate_all()
    
    # Save report
    validator.save_report(results)
    
    # Generate plots if requested
    if args.plot:
        if args.ticker:
            validator.plot_predictions(args.ticker)
        else:
            # Plot first ticker
            first_ticker = validator.model_info['tickers'][0]
            validator.plot_predictions(first_ticker)


if __name__ == "__main__":
    main()