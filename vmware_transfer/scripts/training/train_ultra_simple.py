#!/usr/bin/env python3
"""
# COMPONENT: Ultra-Simple Single-Ticker Training Script
# PURPOSE: Train transformer model on individual stock with CLI ticker selection
# INPUTS: --ticker argument specifying stock symbol, data from data/raw/{ticker}/
# OUTPUTS: Trained model saved as models/model_{ticker}_best.pt, scaler as scalers/scaler_{ticker}.json
# VERIFICATION: Validates ticker existence, tracks RMSE, logs to W&B with ticker tags
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import json
import wandb
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


class TickerDataProcessor:
    """
    # COMPONENT: Data Processor for Single Ticker
    # PURPOSE: Load, validate, and preprocess ticker data with proper scaling
    # INPUTS: Ticker symbol, data directory path
    # OUTPUTS: Normalized sequences, targets, and scaler parameters
    # VERIFICATION: Checks for NaN/Inf, validates shapes, saves scaler stats
    """
    
    def __init__(self, ticker: str, data_dir: Path = Path("data/raw")):
        self.ticker = ticker.upper()
        self.data_dir = data_dir
        self.scaler_params = {}
        self.feature_names = []
        
    def validate_ticker(self) -> bool:
        """Check if ticker data exists"""
        ticker_dir = self.data_dir / self.ticker
        if not ticker_dir.exists():
            return False
        parquet_files = list(ticker_dir.glob("*.parquet"))
        return len(parquet_files) > 0
    
    def load_ticker_data(self) -> pd.DataFrame:
        """Load all parquet files for the ticker"""
        ticker_dir = self.data_dir / self.ticker
        parquet_files = list(ticker_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for ticker {self.ticker}")
        
        # Load the most recent file (sorted by name)
        parquet_files.sort()
        df = pd.read_parquet(parquet_files[-1])
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Engineer features with validation
        Returns feature array and stores feature names
        """
        features = []
        self.feature_names = []
        
        # Price features
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                features.append(df[col].values)
                self.feature_names.append(col)
        
        # Volume (log-transformed to handle scale)
        if 'Volume' in df.columns:
            volume = df['Volume'].values
            # Add small epsilon to avoid log(0)
            log_volume = np.log1p(volume)
            features.append(log_volume)
            self.feature_names.append('LogVolume')
        
        # Technical indicators
        close = df['Close'].values
        
        # Returns
        returns = np.zeros_like(close)
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
        features.append(returns)
        self.feature_names.append('Returns')
        
        # Moving averages
        for window in [5, 10, 20]:
            ma = pd.Series(close).rolling(window=window, min_periods=1).mean().values
            features.append(ma)
            self.feature_names.append(f'MA_{window}')
        
        # Volatility (rolling std of returns)
        volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
        features.append(volatility)
        self.feature_names.append('Volatility')
        
        # Stack features
        feature_array = np.column_stack(features).astype(np.float32)
        
        # Check for NaN or Inf
        if np.any(np.isnan(feature_array)):
            nan_mask = np.isnan(feature_array).any(axis=1)
            logging.warning(f"Found {nan_mask.sum()} rows with NaN values, removing...")
            feature_array = feature_array[~nan_mask]
        
        if np.any(np.isinf(feature_array)):
            inf_mask = np.isinf(feature_array).any(axis=1)
            logging.warning(f"Found {inf_mask.sum()} rows with Inf values, removing...")
            feature_array = feature_array[~inf_mask]
        
        return feature_array
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features with per-feature statistics
        Stores scaler parameters for inference
        """
        normalized = np.zeros_like(features)
        
        for i, feature_name in enumerate(self.feature_names):
            col = features[:, i]
            
            # Calculate statistics
            mean = float(np.mean(col))
            std = float(np.std(col))
            
            # Avoid division by zero
            if std < 1e-8:
                std = 1.0
            
            # Normalize
            normalized[:, i] = (col - mean) / std
            
            # Store scaler parameters
            self.scaler_params[feature_name] = {
                'mean': mean,
                'std': std,
                'min': float(np.min(col)),
                'max': float(np.max(col))
            }
        
        # Verify no NaN/Inf after normalization
        assert not np.any(np.isnan(normalized)), "NaN values after normalization"
        assert not np.any(np.isinf(normalized)), "Inf values after normalization"
        
        return normalized
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        seq_len: int = 60, 
        horizon: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with validation
        Uses percentage returns as targets for scale normalization
        """
        sequences = []
        targets = []
        
        # Get close price index (usually index 3)
        close_idx = self.feature_names.index('Close')
        
        for i in range(seq_len, len(features) - horizon):
            # Input sequence
            seq = features[i-seq_len:i]
            sequences.append(seq)
            
            # Target: percentage change from current to future close prices
            current_close = features[i-1, close_idx]
            future_closes = features[i:i+horizon, close_idx]
            
            # Calculate percentage returns for targets
            pct_returns = (future_closes - current_close) / (current_close + 1e-8)
            targets.append(pct_returns)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        # Validate shapes
        assert sequences.shape[0] == targets.shape[0], "Sequence/target count mismatch"
        assert sequences.shape[1] == seq_len, f"Expected seq_len {seq_len}, got {sequences.shape[1]}"
        assert sequences.shape[2] == len(self.feature_names), "Feature dimension mismatch"
        assert targets.shape[1] == horizon, f"Expected horizon {horizon}, got {targets.shape[1]}"
        
        logging.info(f"Created {len(sequences)} sequences")
        logging.info(f"Sequence shape: {sequences.shape}")
        logging.info(f"Target shape: {targets.shape}")
        
        return sequences, targets
    
    def save_scaler(self, save_dir: Path):
        """Save scaler parameters for inference"""
        save_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = save_dir / f"scaler_{self.ticker}.json"
        
        scaler_data = {
            'ticker': self.ticker,
            'feature_names': self.feature_names,
            'scaler_params': self.scaler_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        logging.info(f"Saved scaler to {scaler_path}")


class ModelTrainer:
    """
    # COMPONENT: Production Model Trainer
    # PURPOSE: Train transformer with proper validation and monitoring
    # INPUTS: Model, data loaders, training configuration
    # OUTPUTS: Trained model weights, training metrics
    # VERIFICATION: Tracks loss convergence, validates on holdout, checks for NaN
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ticker: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.ticker = ticker
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6
        )
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with gradient monitoring"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Ensure prediction shape matches target shape
            if predictions.shape != targets.shape:
                predictions = predictions[:, :targets.shape[1]]
            
            # Calculate loss (MSE for regression)
            loss = nn.MSELoss()(predictions, targets)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"NaN/Inf loss detected at batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log gradient norms periodically
            if batch_idx % 10 == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm > 100:
                    logging.warning(f"Large gradient norm: {total_norm:.2f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate with comprehensive metrics"""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Ensure shape match
                if predictions.shape != targets.shape:
                    predictions = predictions[:, :targets.shape[1]]
                
                # Calculate metrics
                loss = nn.MSELoss()(predictions, targets)
                mse = loss.item()
                mae = nn.L1Loss()(predictions, targets).item()
                
                total_loss += loss.item()
                total_mse += mse
                total_mae += mae
                num_batches += 1
                
                # Store for additional metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Aggregate predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate RMSE
        rmse = np.sqrt(total_mse / num_batches)
        
        # Calculate direction accuracy (for classification metric)
        pred_direction = (all_predictions > 0).float()
        true_direction = (all_targets > 0).float()
        direction_accuracy = (pred_direction == true_direction).float().mean().item()
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_rmse': rmse,
            'val_mae': total_mae / num_batches,
            'val_direction_accuracy': direction_accuracy
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, save_dir: Path):
        """Save model checkpoint with metadata"""
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"model_{self.ticker}_best.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'model_config': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")


def setup_logging(ticker: str):
    """Setup logging with ticker-specific log file"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{ticker}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def main():
    """
    # COMPONENT: Main Training Pipeline
    # PURPOSE: Orchestrate complete training workflow with error handling
    # INPUTS: CLI arguments including ticker symbol
    # OUTPUTS: Trained model, scaler, and comprehensive logs
    # VERIFICATION: Validates all inputs, handles errors gracefully, logs to W&B
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Ultra-Simple Single-Ticker Training')
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=60,
                        help='Sequence length')
    parser.add_argument('--horizon', type=int, default=3,
                        help='Prediction horizon')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension of transformer')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Setup logging
    ticker = args.ticker.upper()
    log_file = setup_logging(ticker)
    logging.info(f"Starting training for ticker: {ticker}")
    logging.info(f"Arguments: {vars(args)}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project="timeseries-transformer",
            name=f"{ticker}_ultra_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            tags=[ticker, "ultra_simple", "single_ticker"]
        )
    
    try:
        # Initialize data processor
        processor = TickerDataProcessor(ticker)
        
        # Validate ticker
        if not processor.validate_ticker():
            available_tickers = [d.name for d in Path("data/raw").iterdir() if d.is_dir()]
            raise ValueError(f"Ticker {ticker} not found. Available: {available_tickers}")
        
        logging.info(f"Loading data for {ticker}...")
        
        # Load and process data
        df = processor.load_ticker_data()
        logging.info(f"Loaded {len(df)} rows of data")
        
        # Engineer features
        features = processor.engineer_features(df)
        logging.info(f"Engineered {features.shape[1]} features")
        
        # Normalize features
        normalized_features = processor.normalize_features(features)
        
        # Create sequences
        sequences, targets = processor.create_sequences(
            normalized_features,
            seq_len=args.seq_len,
            horizon=args.horizon
        )
        
        # Verify no NaN/Inf in data
        assert not np.any(np.isnan(sequences)), "NaN in sequences"
        assert not np.any(np.isinf(sequences)), "Inf in sequences"
        assert not np.any(np.isnan(targets)), "NaN in targets"
        assert not np.any(np.isinf(targets)), "Inf in targets"
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(sequences)
        targets_tensor = torch.FloatTensor(targets)
        
        # Create dataset
        dataset = TensorDataset(sequences_tensor, targets_tensor)
        
        # Split into train/val
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logging.info(f"Train samples: {train_size}, Val samples: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize model
        model = TimeSeriesTransformer(
            input_dim=len(processor.feature_names),
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_length=args.seq_len,
            output_dim=args.horizon,
            forecast_horizon=args.horizon,
            use_attention_pooling=True
        )
        
        # Log model info
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        logging.info(f"Model parameters: {model_info.get('total_parameters', 'Unknown')}")
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            device=device,
            ticker=ticker,
            learning_rate=args.learning_rate
        )
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(args.epochs):
            logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Update scheduler
            trainer.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            current_lr = trainer.optimizer.param_groups[0]['lr']
            logging.info(f"Train Loss: {train_loss:.6f}")
            logging.info(f"Val RMSE: {val_metrics['val_rmse']:.6f}")
            logging.info(f"Val MAE: {val_metrics['val_mae']:.6f}")
            logging.info(f"Val Direction Acc: {val_metrics['val_direction_accuracy']:.2%}")
            logging.info(f"Learning Rate: {current_lr:.2e}")
            
            # Log to W&B
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_metrics['val_loss'],
                    'val_rmse': val_metrics['val_rmse'],
                    'val_mae': val_metrics['val_mae'],
                    'val_direction_accuracy': val_metrics['val_direction_accuracy'],
                    'learning_rate': current_lr
                })
            
            # Save best model
            if val_metrics['val_rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['val_rmse']
                patience_counter = 0
                trainer.save_checkpoint(
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    save_dir=Path("models")
                )
                logging.info(f"New best model! RMSE: {best_val_rmse:.6f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save scaler
        processor.save_scaler(Path("models/scalers"))
        
        # Final summary
        logging.info("\n" + "="*50)
        logging.info(f"Training completed for {ticker}")
        logging.info(f"Best Val RMSE: {best_val_rmse:.6f}")
        logging.info(f"Model saved to: models/model_{ticker}_best.pt")
        logging.info(f"Scaler saved to: models/scalers/scaler_{ticker}.json")
        logging.info(f"Log saved to: {log_file}")
        
        if args.use_wandb:
            wandb.summary['best_val_rmse'] = best_val_rmse
            wandb.finish()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()