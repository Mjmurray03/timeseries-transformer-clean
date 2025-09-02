"""
Extended Model Retraining Pipeline
Fixes overfitting by using 5 years of data with proper validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

# Try to import ta library
try:
    import ta
except ImportError:
    print("Installing technical analysis library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    import ta

# Try to import wandb
try:
    import wandb
    USE_WANDB = True
except ImportError:
    print("Warning: wandb not installed. Proceeding without experiment tracking.")
    print("Install with: pip install wandb")
    USE_WANDB = False

# Import your existing model architecture
sys.path.append(str(Path(__file__).parent.parent))
from src.models.timeseries_transformer import TimeSeriesTransformer


class BalancedDirectionalLoss(nn.Module):
    """Custom loss that penalizes extreme predictions and encourages balanced directions"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Penalty for extreme predictions (>10% in 3 days)
        extreme_penalty = torch.mean(torch.relu(torch.abs(predictions) - 0.10) ** 2)
        
        # Penalty for all predictions having same direction
        pred_signs = torch.sign(predictions)
        direction_variance = torch.var(pred_signs)
        uniformity_penalty = torch.exp(-direction_variance * 2)  # High when all same sign
        
        total_loss = mse_loss + 0.1 * extreme_penalty + 0.05 * uniformity_penalty
        
        return total_loss, {
            'mse': mse_loss.item(),
            'extreme_penalty': extreme_penalty.item(),
            'uniformity_penalty': uniformity_penalty.item()
        }


def download_extended_data(tickers, years=5):
    """Download 5 years of data with progress tracking"""
    print(f"Downloading {years} years of data for {len(tickers)} stocks...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    all_data = {}
    for ticker in tqdm(tickers, desc="Downloading"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[ticker] = df
                print(f"✓ {ticker}: {len(df)} days of data")
            else:
                print(f"✗ {ticker}: No data available")
        except Exception as e:
            print(f"✗ {ticker}: Error - {e}")
    
    return all_data


def add_market_regime_features(df):
    """Add bull/bear/sideways market labels and additional features"""
    # Ensure Close is a Series
    close_prices = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
    
    # Moving averages for regime detection
    df['MA_50'] = close_prices.rolling(window=50).mean()
    df['MA_200'] = close_prices.rolling(window=200).mean()
    
    # Market regime classification
    df['trend_strength'] = (df['MA_50'] - df['MA_200']) / df['MA_200']
    
    # Bull: MA50 > MA200 by at least 2%
    # Bear: MA50 < MA200 by at least 2%
    # Sideways: Otherwise
    df['market_regime'] = 'sideways'
    df.loc[df['trend_strength'] > 0.02, 'market_regime'] = 'bull'
    df.loc[df['trend_strength'] < -0.02, 'market_regime'] = 'bear'
    
    # Distance from MA200 (normalized)
    df['distance_from_MA200'] = (close_prices - df['MA_200']) / df['MA_200']
    
    return df


def create_enhanced_features(df):
    """Create comprehensive feature set with technical indicators"""
    features_df = df.copy()
    
    # Ensure Close is a Series (1D), not DataFrame
    close_prices = features_df['Close'].squeeze() if hasattr(features_df['Close'], 'squeeze') else features_df['Close']
    
    # Price features
    features_df['returns'] = close_prices.pct_change()
    features_df['log_returns'] = np.log(close_prices / close_prices.shift(1))
    features_df['volatility_20d'] = features_df['returns'].rolling(window=20).std()
    
    # Technical indicators using ta library - ensure Series input
    features_df['RSI'] = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=close_prices)
    features_df['MACD'] = macd.macd()
    features_df['MACD_signal'] = macd.macd_signal()
    features_df['MACD_diff'] = macd.macd_diff()
    
    # Bollinger Bands position
    bb = ta.volatility.BollingerBands(close=close_prices)
    features_df['BB_position'] = (close_prices - bb.bollinger_mavg()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )
    
    # Volume features - ensure Series
    volume = features_df['Volume'].squeeze() if hasattr(features_df['Volume'], 'squeeze') else features_df['Volume']
    features_df['volume_ratio'] = volume / volume.rolling(window=20).mean()
    
    # Temporal features
    features_df['day_of_week'] = pd.to_datetime(features_df.index).dayofweek
    features_df['month'] = pd.to_datetime(features_df.index).month
    features_df['quarter'] = pd.to_datetime(features_df.index).quarter
    
    # Add market regime features
    features_df = add_market_regime_features(features_df)
    
    # Drop NaN values from feature engineering
    features_df = features_df.dropna()
    
    return features_df


def create_balanced_dataset(data_dict, val_split=0.15, test_split=0.15, gap_days=60):
    """Create train/val/test sets with temporal ordering and gap"""
    
    all_features = []
    all_targets = []
    all_tickers = []
    
    for ticker, df in data_dict.items():
        # Create enhanced features
        features_df = create_enhanced_features(df)
        
        # Select feature columns (excluding target and non-numeric)
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'returns', 'log_returns', 'volatility_20d',
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_position', 'volume_ratio',
            'MA_50', 'MA_200', 'trend_strength', 'distance_from_MA200',
            'day_of_week', 'month', 'quarter'
        ]
        
        # Prepare sequences
        seq_len = 60
        forecast_horizon = 3
        
        for i in range(seq_len, len(features_df) - forecast_horizon):
            # Input sequence
            seq_features = features_df[feature_cols].iloc[i-seq_len:i].values
            
            # Target: next 3 days returns
            close_prices = features_df['Close'].squeeze() if hasattr(features_df['Close'], 'squeeze') else features_df['Close']
            future_prices = close_prices.iloc[i:i+forecast_horizon].values
            current_price = close_prices.iloc[i-1]
            target_returns = (future_prices - current_price) / current_price
            
            all_features.append(seq_features)
            all_targets.append(target_returns)
            all_tickers.append(ticker)
    
    # Convert to arrays
    all_features = np.array(all_features)
    all_targets = np.array(all_targets)
    
    # Calculate split indices
    n_samples = len(all_features)
    train_end = int(n_samples * (1 - val_split - test_split))
    val_end = int(n_samples * (1 - test_split))
    
    # Apply temporal gap between sets
    train_end -= gap_days
    val_start = train_end + gap_days
    val_end = val_start + int(n_samples * val_split)
    test_start = val_end + gap_days
    
    # Create splits
    train_data = {
        'features': all_features[:train_end],
        'targets': all_targets[:train_end],
        'tickers': all_tickers[:train_end]
    }
    
    val_data = {
        'features': all_features[val_start:val_end],
        'targets': all_targets[val_start:val_end],
        'tickers': all_tickers[val_start:val_end]
    }
    
    test_data = {
        'features': all_features[test_start:],
        'targets': all_targets[test_start:],
        'tickers': all_tickers[test_start:]
    }
    
    return train_data, val_data, test_data


def validate_predictions_realistic(predictions, phase="validation"):
    """Check if predictions fall within realistic ranges"""
    
    predictions_np = predictions.detach().cpu().numpy()
    
    # Calculate statistics
    avg_magnitude = np.abs(predictions_np).mean()
    max_magnitude = np.abs(predictions_np).max()
    std_dev = predictions_np.std()
    
    # Count extreme predictions
    extreme_count = np.sum(np.abs(predictions_np) > 0.10)  # >10% in 3 days
    extreme_percentage = (extreme_count / predictions_np.size) * 100
    
    # Check directional balance
    bullish_count = np.sum(predictions_np > 0)
    bearish_count = np.sum(predictions_np < 0)
    directional_ratio = bullish_count / (bullish_count + bearish_count + 1e-8)
    
    warnings_list = []
    
    if avg_magnitude > 0.05:  # >5% average is suspicious
        warnings_list.append(f"High average magnitude: {avg_magnitude:.4f}")
    
    if extreme_percentage > 5:  # More than 5% extreme predictions
        warnings_list.append(f"Too many extreme predictions: {extreme_percentage:.1f}%")
    
    if directional_ratio < 0.2 or directional_ratio > 0.8:  # Too biased
        warnings_list.append(f"Directional imbalance: {directional_ratio:.2f}")
    
    # Print validation report
    print(f"\n=== {phase.upper()} Prediction Validation ===")
    print(f"Average magnitude: {avg_magnitude:.4f} ({avg_magnitude*100:.2f}%)")
    print(f"Max magnitude: {max_magnitude:.4f} ({max_magnitude*100:.2f}%)")
    print(f"Std deviation: {std_dev:.4f}")
    print(f"Extreme predictions (>10%): {extreme_percentage:.1f}%")
    print(f"Bullish/Total ratio: {directional_ratio:.2f}")
    
    if warnings_list:
        print("\n⚠️ WARNINGS:")
        for warning in warnings_list:
            print(f"  - {warning}")
        return False
    else:
        print("✅ Predictions appear realistic")
        return True


def train_with_validation(model, train_data, val_data, epochs=20, lr=1e-3, batch_size=32, device='cuda'):
    """Training loop with early stopping and validation checks"""
    
    # Initialize wandb if available
    if USE_WANDB:
        wandb.init(
            project="timeseries-transformer-extended",
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "architecture": "TimeSeriesTransformer",
                "loss": "BalancedDirectional"
            }
        )
    
    # Prepare data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data['features']),
        torch.FloatTensor(train_data['targets'])
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_data['features']),
        torch.FloatTensor(val_data['targets'])
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = BalancedDirectionalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss, loss_components = criterion(predictions, batch_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_predictions.append(predictions.detach())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions = model(batch_features)
                loss, _ = criterion(predictions, batch_targets)
                
                val_losses.append(loss.item())
                val_predictions.append(predictions)
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            all_val_preds = torch.cat(val_predictions, dim=0)
            validate_predictions_realistic(all_val_preds, f"Epoch {epoch+1}")
        
        # Log metrics
        if USE_WANDB:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': current_lr,
                'epoch': epoch
            })
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'history': history
            }, 'model_extended_best.pt')
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    if USE_WANDB:
        wandb.finish()
    return model, history


def create_diagnostic_plots(history, predictions, actuals, save_dir='plots'):
    """Generate diagnostic plots"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Prediction distribution
    axes[0, 1].hist(predictions.flatten(), bins=50, alpha=0.7, label='Predictions')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[0, 1].set_xlabel('Predicted Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Distribution')
    axes[0, 1].legend()
    
    # Scatter plot: Predicted vs Actual
    axes[1, 0].scatter(actuals.flatten(), predictions.flatten(), alpha=0.5, s=1)
    axes[1, 0].plot([-0.2, 0.2], [-0.2, 0.2], 'r--', label='Perfect Prediction')
    axes[1, 0].set_xlabel('Actual Returns')
    axes[1, 0].set_ylabel('Predicted Returns')
    axes[1, 0].set_title('Predicted vs Actual Returns')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate schedule
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_diagnostics.png", dpi=150)
    plt.show()
    
    print(f"Diagnostic plots saved to {save_dir}/")


def main():
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']
    YEARS = 5
    EPOCHS = 20  # Minimum 20 epochs as specified
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    print("="*50)
    
    # Step 1: Download extended data
    data_dict = download_extended_data(TICKERS, years=YEARS)
    
    if len(data_dict) == 0:
        print("Error: No data downloaded")
        return
    
    # Step 2: Create balanced dataset with enhanced features
    print("\nCreating enhanced feature sets...")
    train_data, val_data, test_data = create_balanced_dataset(data_dict)
    
    print(f"Train samples: {len(train_data['features'])}")
    print(f"Val samples: {len(val_data['features'])}")
    print(f"Test samples: {len(test_data['features'])}")
    
    # Step 3: Initialize model
    input_dim = train_data['features'].shape[-1]  # Number of features
    print(f"\nModel input dimension: {input_dim}")
    
    # Try to create model with dropout, fall back if not supported
    try:
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            hidden_dim=256,  # Increased from 128
            num_heads=8,     # Increased from 4
            num_layers=4,    # Increased from 2
            forecast_horizon=3,
            output_dim=3,
            dropout=0.1
        )
    except TypeError:
        # Fallback if dropout parameter not supported
        print("Note: Model doesn't support dropout parameter, creating without it")
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            hidden_dim=256,  # Increased from 128
            num_heads=8,     # Increased from 4
            num_layers=4,    # Increased from 2
            forecast_horizon=3,
            output_dim=3
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Step 4: Train with validation
    print("\nStarting training with balanced loss...")
    model, history = train_with_validation(
        model, train_data, val_data,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # Step 5: Final evaluation on test set
    print("\n=== FINAL TEST SET EVALUATION ===")
    model.to(DEVICE)
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data['features']),
        torch.FloatTensor(test_data['targets'])
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(DEVICE)
            predictions = model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(targets.numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_actuals = np.concatenate(all_actuals)
    
    # Validate test predictions
    is_realistic = validate_predictions_realistic(
        torch.tensor(all_predictions), 
        phase="TEST SET"
    )
    
    # Calculate final metrics
    mse = np.mean((all_predictions - all_actuals) ** 2)
    rmse = np.sqrt(mse)
    directional_accuracy = np.mean(
        np.sign(all_predictions) == np.sign(all_actuals)
    )
    
    print(f"\nFinal Test Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Directional Accuracy: {directional_accuracy:.2%}")
    
    # Step 6: Create diagnostic plots
    create_diagnostic_plots(history, all_predictions, all_actuals)
    
    # Step 7: Save final model and scalers
    final_save = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'forecast_horizon': 3
        },
        'performance_metrics': {
            'test_rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'is_realistic': is_realistic
        },
        'training_history': history,
        'tickers': TICKERS
    }
    
    torch.save(final_save, 'model_extended_final.pt')
    
    # Save scaler information
    scaler_info = {
        'feature_columns': [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'returns', 'log_returns', 'volatility_20d',
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_position', 'volume_ratio',
            'MA_50', 'MA_200', 'trend_strength', 'distance_from_MA200',
            'day_of_week', 'month', 'quarter'
        ],
        'seq_len': 60,
        'forecast_horizon': 3
    }
    
    with open('scaler_config.json', 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Model saved as: model_extended_final.pt")
    print(f"Best checkpoint saved as: model_extended_best.pt")
    print(f"Predictions realistic: {'✅ YES' if is_realistic else '❌ NO'}")
    print("="*50)


if __name__ == "__main__":
    main()