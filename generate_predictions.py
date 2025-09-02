#!/usr/bin/env python3
"""
Phase 5 Step 3: Generate Price Predictions
Purpose: Generate actual price predictions for tomorrow/next periods
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


class PricePredictionGenerator:
    """Generate price predictions from trained models"""
    
    def __init__(self, model_path: str, data_dir: str = "data/raw", scalers_dir: str = "scalers"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.scalers_dir = Path(scalers_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and info
        self.checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model_info = self._extract_model_info()
        self.model = self._load_model()
        self.scalers = self._load_scalers()
        
        print(f"Loaded model: {self.model_path.name}")
        print(f"Model type: {self.model_info['type']}")
        print(f"Tickers: {', '.join(self.model_info['tickers'])}")
        print(f"Device: {self.device}")
        print("="*60)
    
    def _extract_model_info(self) -> Dict:
        """Extract model information"""
        info = {
            'config': self.checkpoint.get('config', {}),
            'seq_len': self.checkpoint.get('config', {}).get('seq_len', 60),
            'horizon': self.checkpoint.get('config', {}).get('horizon', 3)
        }
        
        if 'tickers' in self.checkpoint:
            info['type'] = 'multi_stock'
            info['tickers'] = self.checkpoint['tickers']
        elif 'ticker' in self.checkpoint:
            info['type'] = 'single_stock'
            info['tickers'] = [self.checkpoint['ticker']]
        else:
            # Infer from filename
            if 'multi' in str(self.model_path):
                info['type'] = 'multi_stock'
                if '8stocks' in str(self.model_path):
                    info['tickers'] = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NFLX', 'NVDA', 'TSLA']
                elif '3stocks' in str(self.model_path):
                    info['tickers'] = ['AAPL', 'MSFT', 'NVDA']
            else:
                info['type'] = 'single_stock'
                parts = str(self.model_path.name).split('_')
                ticker = parts[1] if len(parts) > 1 else 'AAPL'
                info['tickers'] = [ticker]
        
        return info
    
    def _load_model(self) -> nn.Module:
        """Load the trained model"""
        config = self.model_info['config']
        
        model = TimeSeriesTransformer(
            input_dim=10,  # Fixed feature set
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            output_dim=self.model_info['horizon'],
            dropout=0  # No dropout for inference
        ).to(self.device)
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _load_scalers(self) -> Dict:
        """Load normalization parameters"""
        scalers = {}
        for ticker in self.model_info['tickers']:
            scaler_path = self.scalers_dir / f"scaler_{ticker}.json"
            if scaler_path.exists():
                with open(scaler_path, 'r') as f:
                    scalers[ticker] = json.load(f)
        return scalers
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features from raw data"""
        features = []
        
        # Basic OHLCV
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
    
    def predict_ticker(self, ticker: str, num_monte_carlo: int = 10) -> Dict:
        """Generate predictions for a single ticker with uncertainty estimates"""
        
        # Load data
        data_file = self.data_dir / f"{ticker}.parquet"
        if not data_file.exists():
            data_file = self.data_dir / f"{ticker.lower()}.parquet"
        
        df = pd.read_parquet(data_file)
        df = df.sort_index()
        
        # Get latest data
        latest_date = df.index[-1]
        latest_price = df['Close'].iloc[-1]
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Normalize
        if ticker in self.scalers:
            scaler = self.scalers[ticker]
            mean = np.array(scaler['mean'])
            std = np.array(scaler['std'])
            normalized = (features - mean) / std
        else:
            print(f"Warning: No scaler for {ticker}, using local normalization")
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            normalized = (features - mean) / std
        
        # Get last sequence
        seq_len = self.model_info['seq_len']
        if len(normalized) < seq_len:
            return {'error': f'Insufficient data for {ticker}'}
        
        last_sequence = normalized[-seq_len:]
        
        # Run multiple predictions with dropout for uncertainty
        predictions = []
        
        # First, get deterministic prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            base_prediction = self.model(input_tensor).cpu().numpy()[0]
        
        # Monte Carlo dropout for uncertainty (if model supports it)
        if num_monte_carlo > 1:
            # Enable dropout for uncertainty estimation
            def enable_dropout(model):
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
            
            enable_dropout(self.model)
            
            for _ in range(num_monte_carlo):
                with torch.no_grad():
                    pred = self.model(input_tensor).cpu().numpy()[0]
                    predictions.append(pred)
            
            self.model.eval()  # Reset to eval mode
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
        else:
            mean_pred = base_prediction
            std_pred = np.zeros_like(base_prediction)
        
        # Denormalize predictions (for close price, index 3)
        if ticker in self.scalers:
            close_mean = scaler['mean'][3]
            close_std = scaler['std'][3]
            
            price_predictions = mean_pred * close_std + close_mean
            price_uncertainty = std_pred * close_std
        else:
            # Fallback
            price_predictions = mean_pred * std[3] + mean[3]
            price_uncertainty = std_pred * std[3]
        
        # Calculate percentage changes
        pct_changes = ((price_predictions - latest_price) / latest_price) * 100
        
        # Generate future dates (trading days only)
        future_dates = []
        current_date = latest_date
        days_added = 0
        while days_added < self.model_info['horizon']:
            current_date = current_date + timedelta(days=1)
            # Skip weekends (simplified - doesn't account for holidays)
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                future_dates.append(current_date)
                days_added += 1
        
        return {
            'ticker': ticker,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'latest_price': float(latest_price),
            'predictions': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'prices': price_predictions.tolist(),
                'price_lower': (price_predictions - 2*price_uncertainty).tolist(),
                'price_upper': (price_predictions + 2*price_uncertainty).tolist(),
                'pct_changes': pct_changes.tolist(),
                'confidence_95': (2*price_uncertainty).tolist()
            },
            'horizon_days': self.model_info['horizon'],
            'model_confidence': 'high' if std_pred.mean() < 0.1 else 'medium' if std_pred.mean() < 0.2 else 'low'
        }
    
    def generate_all_predictions(self, output_format: str = 'detailed') -> Dict:
        """Generate predictions for all tickers"""
        
        all_predictions = {
            'generation_time': datetime.now().isoformat(),
            'model': str(self.model_path.name),
            'model_type': self.model_info['type'],
            'predictions': {}
        }
        
        print("\nGENERATING PREDICTIONS")
        print("="*60)
        
        for ticker in self.model_info['tickers']:
            print(f"\nProcessing {ticker}...")
            
            try:
                prediction = self.predict_ticker(ticker)
                
                if 'error' not in prediction:
                    all_predictions['predictions'][ticker] = prediction
                    
                    # Print summary
                    print(f"  Current Price: ${prediction['latest_price']:.2f}")
                    print(f"  Predictions for next {prediction['horizon_days']} days:")
                    
                    for i, (date, price, pct) in enumerate(zip(
                        prediction['predictions']['dates'],
                        prediction['predictions']['prices'],
                        prediction['predictions']['pct_changes']
                    )):
                        direction = "↑" if pct > 0 else "↓"
                        print(f"    {date}: ${price:.2f} ({direction} {abs(pct):.2f}%)")
                    
                    print(f"  Model Confidence: {prediction['model_confidence']}")
                else:
                    print(f"  Error: {prediction['error']}")
                    
            except Exception as e:
                print(f"  Failed to generate prediction: {str(e)}")
        
        return all_predictions
    
    def save_predictions(self, predictions: Dict, output_dir: str = "predictions"):
        """Save predictions to file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"predictions_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save CSV summary
        csv_path = output_dir / f"predictions_summary_{timestamp}.csv"
        
        rows = []
        for ticker, pred in predictions['predictions'].items():
            for i in range(len(pred['predictions']['dates'])):
                rows.append({
                    'ticker': ticker,
                    'current_price': pred['latest_price'],
                    'prediction_date': pred['predictions']['dates'][i],
                    'predicted_price': pred['predictions']['prices'][i],
                    'price_lower_95': pred['predictions']['price_lower'][i],
                    'price_upper_95': pred['predictions']['price_upper'][i],
                    'pct_change': pred['predictions']['pct_changes'][i],
                    'confidence': pred['model_confidence']
                })
        
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(csv_path, index=False)
        
        print(f"\nPredictions saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return json_path, csv_path
    
    def generate_trading_signals(self, predictions: Dict, threshold: float = 2.0) -> pd.DataFrame:
        """Generate simple trading signals based on predictions"""
        
        signals = []
        
        for ticker, pred in predictions['predictions'].items():
            # Look at first prediction (tomorrow)
            if pred['predictions']['prices']:
                tomorrow_pct = pred['predictions']['pct_changes'][0]
                
                if tomorrow_pct > threshold:
                    signal = 'STRONG BUY'
                elif tomorrow_pct > 0:
                    signal = 'BUY'
                elif tomorrow_pct < -threshold:
                    signal = 'STRONG SELL'
                elif tomorrow_pct < 0:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals.append({
                    'Ticker': ticker,
                    'Current': f"${pred['latest_price']:.2f}",
                    'Predicted': f"${pred['predictions']['prices'][0]:.2f}",
                    'Change': f"{tomorrow_pct:+.2f}%",
                    'Signal': signal,
                    'Confidence': pred['model_confidence'].upper()
                })
        
        df_signals = pd.DataFrame(signals)
        
        print("\n" + "="*60)
        print("TRADING SIGNALS (Tomorrow)")
        print("="*60)
        print(df_signals.to_string(index=False))
        print("="*60)
        print("Disclaimer: These are ML model predictions, not financial advice!")
        
        return df_signals


def main():
    parser = argparse.ArgumentParser(description='Generate price predictions')
    parser.add_argument('--model', type=str, default='models/model_multi_8stocks_best.pt',
                        help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Data directory')
    parser.add_argument('--scalers-dir', type=str, default='scalers',
                        help='Scalers directory')
    parser.add_argument('--save', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--monte-carlo', type=int, default=1,
                        help='Number of Monte Carlo runs for uncertainty (1=deterministic)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PricePredictionGenerator(
        model_path=args.model,
        data_dir=args.data_dir,
        scalers_dir=args.scalers_dir
    )
    
    # Generate predictions
    predictions = predictor.generate_all_predictions()
    
    # Save if requested
    if args.save:
        predictor.save_predictions(predictions)
    
    # Generate trading signals
    signals = predictor.generate_trading_signals(predictions)
    
    # Risk warning
    print("  IMPORTANT DISCLAIMER ")
    print("These predictions are from a machine learning model trained on historical data.")
    print("They should NOT be used as the sole basis for investment decisions.")
    print("Always do your own research and consult with financial professionals.")


if __name__ == "__main__":
    main()