#!/usr/bin/env python3
"""
Ultra-Simple Single-Ticker Training Script

Train a transformer model on individual stock data with automatic GPU detection and flexible configuration.

Features:
- Automatic GPU detection with CPU fallback
- Optional configuration file support
- Flexible ticker selection
- Comprehensive logging and metrics

Usage Examples:
    # Train on AAPL with default settings
    python train_ultra_simple.py --ticker AAPL

    # Train with custom parameters
    python train_ultra_simple.py --ticker MSFT --epochs 50 --batch-size 64

    # Train with configuration file
    python train_ultra_simple.py --ticker NVDA --config config.yaml

    # Enable Weights & Biases logging
    python train_ultra_simple.py --ticker TSLA --use-wandb

Inputs:
    --ticker: Stock symbol (required)
    --config: Optional YAML configuration file
    Other parameters: epochs, batch-size, learning-rate, etc.

Outputs:
    - Trained model: models/model_{ticker}_best.pt
    - Scaler parameters: scalers/scaler_{ticker}.json
    - Training logs: logs/training_{ticker}_{timestamp}.log
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb
from torch.utils.data import DataLoader, TensorDataset, random_split

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

    def __init__(
        self, ticker: str, data_dir: Path = Path("data/raw"), scalers_dir: Path = Path("scalers")
    ):
        self.ticker = ticker.upper()
        self.data_dir = Path(data_dir)
        self.scalers_dir = Path(scalers_dir)
        self.scaler_params = {}
        self.feature_names = []

    def discover_available_tickers(self) -> Dict[str, Set[str]]:
        """Discover all available tickers from data and scalers"""
        # Discover data tickers from parquet files
        data_tickers = set()
        if self.data_dir.exists():
            for file in self.data_dir.glob("*.parquet"):
                ticker = file.stem.upper()
                data_tickers.add(ticker)

        # Discover scaler tickers
        scaler_tickers = set()
        if self.scalers_dir.exists():
            for file in self.scalers_dir.glob("scaler_*.json"):
                match = re.match(r"scaler_(.+)\.json", file.name)
                if match:
                    ticker = match.group(1).upper()
                    scaler_tickers.add(ticker)

        return {
            "data": data_tickers,
            "scalers": scaler_tickers,
            "available": data_tickers,  # Use data files as source of truth
        }

    def validate_ticker(self) -> bool:
        """Check if ticker data exists"""
        ticker_file = self.data_dir / f"{self.ticker}.parquet"
        return ticker_file.exists()

    def load_ticker_data(self) -> pd.DataFrame:
        """Load parquet file for the ticker"""
        ticker_file = self.data_dir / f"{self.ticker}.parquet"

        if not ticker_file.exists():
            # Try lowercase
            ticker_file_lower = self.data_dir / f"{self.ticker.lower()}.parquet"
            if ticker_file_lower.exists():
                ticker_file = ticker_file_lower
            else:
                raise FileNotFoundError(f"No parquet file found for ticker {self.ticker}")

        df = pd.read_parquet(ticker_file)

        # Ensure required columns exist
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try lowercase columns
            df.columns = [col.capitalize() for col in df.columns]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by index (assumes datetime index)
        df = df.sort_index()

        return df

    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features from raw OHLCV data"""
        features = []

        # Basic OHLCV features
        features.append(df["Open"].values)
        features.append(df["High"].values)
        features.append(df["Low"].values)
        features.append(df["Close"].values)
        features.append(df["Volume"].values)

        # Technical indicators
        # Returns
        features.append(df["Close"].pct_change().fillna(0).values)

        # Moving averages
        features.append(df["Close"].rolling(5).mean().fillna(df["Close"]).values)
        features.append(df["Close"].rolling(20).mean().fillna(df["Close"]).values)

        # Volatility (rolling std of returns)
        returns = df["Close"].pct_change()
        features.append(returns.rolling(20).std().fillna(0).values)

        # Volume moving average
        features.append(df["Volume"].rolling(5).mean().fillna(df["Volume"]).values)

        # Store feature names for reference
        self.feature_names = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Returns",
            "MA5",
            "MA20",
            "Volatility",
            "Volume_MA5",
        ]

        # Stack features
        features = np.stack(features, axis=1)

        return features

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using standardization"""
        # Calculate statistics
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8  # Avoid division by zero

        # Store scaler parameters
        self.scaler_params = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_names": self.feature_names,
            "ticker": self.ticker,
        }

        # Normalize
        normalized = (features - mean) / std

        return normalized

    def save_scaler(self):
        """Save scaler parameters to JSON"""
        self.scalers_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = self.scalers_dir / f"scaler_{self.ticker}.json"

        with open(scaler_path, "w") as f:
            json.dump(self.scaler_params, f, indent=2)

        logging.info(f"Saved scaler to {scaler_path}")

        return scaler_path

    def create_sequences(
        self, data: np.ndarray, seq_len: int = 60, horizon: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        sequences = []
        targets = []

        for i in range(seq_len, len(data) - horizon):
            sequences.append(data[i - seq_len : i])
            # Predict close price (index 3)
            targets.append(data[i : i + horizon, 3])

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return sequences, targets


class SimpleTrainer:
    """Simple trainer class for single ticker"""

    def __init__(self, model: nn.Module, ticker: str, device: torch.device):
        self.model = model.to(device)
        self.ticker = ticker
        self.device = device
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(sequences)
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(sequences)
                loss = self.criterion(predictions, targets)

                # Calculate metrics
                mse = torch.mean((predictions - targets) ** 2).item()
                mae = torch.mean(torch.abs(predictions - targets)).item()

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
            "val_loss": total_loss / num_batches,
            "val_rmse": rmse,
            "val_mae": total_mae / num_batches,
            "val_direction_accuracy": direction_accuracy,
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, save_dir: Path):
        """Save model checkpoint with metadata"""
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"model_{self.ticker}_best.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "ticker": self.ticker,
            "timestamp": datetime.now().isoformat(),
            "model_config": (
                self.model.get_model_info() if hasattr(self.model, "get_model_info") else {}
            ),
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
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_file


def load_config_from_args_and_file(args, parser) -> Dict:
    """Load configuration from config file or command line arguments."""
    if args.config and os.path.exists(args.config):
        # Load from config file
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        logging.info(f"Loaded configuration from {args.config}")
    else:
        # Use command line arguments - only include attributes that exist
        config = {}
        arg_mappings = {
            'ticker': 'ticker',
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'val_split': 'val_split',
            'seq_len': 'seq_len',
            'horizon': 'horizon',
            'hidden_dim': 'hidden_dim',
            'num_layers': 'num_layers',
            'num_heads': 'num_heads',
            'dropout': 'dropout',
            'use_wandb': 'use_wandb',
            'wandb_project': 'wandb_project',
            'device': 'device',
            'data_dir': 'data_dir',
            'scalers_dir': 'scalers_dir',
            'allow_data_without_scalers': 'allow_data_without_scalers'
        }

        for config_key, attr_name in arg_mappings.items():
            if hasattr(args, attr_name):
                config[config_key] = getattr(args, attr_name)
        if args.config:
            logging.warning(f"Config file {args.config} not found, using command line arguments")
        else:
            logging.info("Using command line arguments (no config file specified)")

    # Override config file values with explicitly provided CLI arguments
    # This ensures CLI args always take precedence
    cli_overrides = {}
    for key, value in vars(args).items():
        # Skip special keys that aren't configuration parameters
        if key in ['config']:
            continue
        # If the argument was explicitly provided (not default), it overrides config
        if parser.get_default(key) != value:
            cli_overrides[key] = value

    # Apply CLI overrides to config
    config.update(cli_overrides)

    # Convert back to argparse namespace for compatibility
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return config


def main():
    """
    Main Training Pipeline

    Orchestrates the complete training workflow with:
    - Automatic GPU detection with CPU fallback
    - Flexible configuration (CLI args or config file)
    - Comprehensive error handling and logging
    - Optional Weights & Biases integration
    """

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train TimeSeries Transformer on Stock Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --ticker AAPL                    # Train on AAPL with defaults
    %(prog)s --ticker MSFT --epochs 50        # Train MSFT for 50 epochs
    %(prog)s --ticker NVDA --config cfg.yaml  # Use configuration file
    %(prog)s --ticker TSLA --use-wandb        # Enable W&B logging
        """
    )

    # Required argument
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL, NVDA, TSLA). "
             "The script will look for data in data/raw/{TICKER}.parquet"
    )

    # Optional configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional). "
             "If provided, config file parameters override defaults but CLI args override config."
    )
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")

    # Data parameters
    parser.add_argument("--seq-len", type=int, default=60, help="Input sequence length in days (default: 60)")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon in days (default: 3)")

    # Model architecture
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden dimension of transformer (default: 128)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4,
        help="Number of transformer layers (default: 4)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate for regularization (default: 0.1)"
    )

    # Logging and monitoring
    parser.add_argument(
        "--use-wandb", action="store_true",
        help="Enable Weights & Biases logging for experiment tracking"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="timeseries-transformer",
        help="W&B project name (default: timeseries-transformer)"
    )

    # NEW: Add directory overrides
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/raw or TST_DATA_DIR env)",
    )
    parser.add_argument(
        "--scalers-dir",
        type=str,
        default=None,
        help="Scalers directory (default: scalers or TST_SCALERS_DIR env)",
    )
    parser.add_argument(
        "--allow-data-without-scalers",
        action="store_true",
        default=True,
        help="Allow training without pre-existing scalers",
    )

    # Device selection
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training. 'auto' will use GPU if available (default: auto)"
    )

    args = parser.parse_args()

    # Load configuration from file or command line arguments
    try:
        config = load_config_from_args_and_file(args, parser)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Resolve directories with env fallbacks
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif os.environ.get("TST_DATA_DIR"):
        data_dir = Path(os.environ["TST_DATA_DIR"])
    else:
        data_dir = Path("data/raw")

    if args.scalers_dir:
        scalers_dir = Path(args.scalers_dir)
    elif os.environ.get("TST_SCALERS_DIR"):
        scalers_dir = Path(os.environ["TST_SCALERS_DIR"])
    else:
        scalers_dir = Path("scalers")

    # Setup logging
    ticker = args.ticker.upper()
    log_file = setup_logging(ticker)

    # Log initial discovery information
    logging.info("=" * 60)
    logging.info("TICKER DISCOVERY DIAGNOSTICS")
    logging.info("=" * 60)
    logging.info(f"Current working directory: {Path.cwd()}")
    logging.info(f"Script file: {Path(__file__).resolve()}")
    logging.info(f"Data directory (resolved): {data_dir.resolve()}")
    logging.info(f"Scalers directory (resolved): {scalers_dir.resolve()}")
    logging.info(f"Data dir exists: {data_dir.exists()}")
    logging.info(f"Scalers dir exists: {scalers_dir.exists()}")

    # Initialize data processor with resolved directories
    processor = TickerDataProcessor(ticker, data_dir=data_dir, scalers_dir=scalers_dir)

    # Discover available tickers
    ticker_info = processor.discover_available_tickers()

    logging.info(f"\nDiscovered tickers from data: {sorted(ticker_info['data'])}")
    logging.info(f"Count: {len(ticker_info['data'])}")
    logging.info(f"\nDiscovered tickers from scalers: {sorted(ticker_info['scalers'])}")
    logging.info(f"Count: {len(ticker_info['scalers'])}")
    logging.info(f"\nFinal available tickers: {sorted(ticker_info['available'])}")
    logging.info(f"Count: {len(ticker_info['available'])}")
    logging.info("=" * 60)

    logging.info(f"Starting training for ticker: {ticker}")
    logging.info(f"Arguments: {vars(args)}")

    # Initialize W&B if requested
    if args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"{ticker}_ultra_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
                tags=[ticker, "ultra_simple", "single_ticker", device.type],
            )
            logging.info(f"Initialized W&B project: {args.wandb_project}")
        except Exception as e:
            logging.warning(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
            args.use_wandb = False

    try:
        # Validate ticker
        if not processor.validate_ticker():
            available = sorted(ticker_info["available"])
            error_msg = (
                f"\nTicker {ticker} not found.\n"
                f"Data directory: {data_dir.resolve()}\n"
                f"Scalers directory: {scalers_dir.resolve()}\n"
                f"Data tickers found: {ticker_info['data']}\n"
                f"Scaler tickers found: {ticker_info['scalers']}\n"
                f"Available tickers: {available}\n"
            )
            raise ValueError(error_msg)

        logging.info(f"Loading data for {ticker}...")

        # Load and process data
        df = processor.load_ticker_data()
        logging.info(f"Loaded {len(df)} rows of data")

        # Engineer features
        features = processor.engineer_features(df)
        logging.info(f"Engineered {features.shape[1]} features")

        # Normalize features
        normalized_features = processor.normalize_features(features)

        # Save scaler
        scaler_path = processor.save_scaler()

        # Create sequences
        sequences, targets = processor.create_sequences(
            normalized_features, seq_len=args.seq_len, horizon=args.horizon
        )

        logging.info(f"Created {len(sequences)} sequences")
        logging.info(f"Sequence shape: {sequences.shape}")
        logging.info(f"Target shape: {targets.shape}")

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
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Device configuration with detailed GPU info
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        # Print device information
        print("\n" + "="*60)
        print("DEVICE CONFIGURATION")
        print("="*60)
        print(f"Using device: {device}")

        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")

            # Set deterministic behavior for reproducibility
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("Running on CPU - Training may be slower")
            print("To use GPU, ensure CUDA is installed and compatible with PyTorch")

        print("="*60 + "\n")

        logging.info(f"Using device: {device}")
        if device.type == "cuda":
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

        model = TimeSeriesTransformer(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=args.horizon,
            dropout=args.dropout,
        )

        # Initialize trainer with device object (not string)
        trainer = SimpleTrainer(model, ticker, device=device)
        trainer.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            logging.info(f"\nEpoch {epoch}/{args.epochs}")

            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = trainer.validate(val_loader)

            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logging.info(f"Val RMSE: {val_metrics['val_rmse']:.4f}")
            logging.info(f"Val Direction Accuracy: {val_metrics['val_direction_accuracy']:.2%}")

            # Log to W&B
            if args.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                trainer.save_checkpoint(epoch, val_metrics, Path("models"))
                patience_counter = 0
                logging.info(f"New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch} epochs")
                    break

        # Final summary
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING COMPLETE")
        logging.info(f"Best Val Loss: {best_val_loss:.4f}")
        logging.info(f"Model saved to: models/model_{ticker}_best.pt")
        logging.info(f"Scaler saved to: {scaler_path}")
        logging.info(f"Log file: {log_file}")
        logging.info("=" * 60)

        # Finish W&B run
        if args.use_wandb:
            wandb.finish()

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
