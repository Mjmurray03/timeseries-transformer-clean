#!/usr/bin/env python3
"""
# COMPONENT: Multi-Stock Training Script (FIXED)
# PURPOSE: Train transformer model on multiple stocks simultaneously
# FIXES: Handles flat parquet files (AAPL.parquet) instead of subdirectories
# INPUTS: Multiple tickers, data from data/raw/{ticker}.parquet files
# OUTPUTS: Single multi-stock model, per-ticker scalers
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
import wandb
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


def discover_all_tickers(data_dir: Path, scalers_dir: Path) -> Dict[str, Set[str]]:
    """Discover all available tickers from data and scalers"""
    # Discover data tickers from parquet files
    data_tickers = set()
    if data_dir.exists():
        for file in data_dir.glob("*.parquet"):
            ticker = file.stem.upper()
            data_tickers.add(ticker)

    # Discover scaler tickers
    scaler_tickers = set()
    if scalers_dir.exists():
        for file in scalers_dir.glob("scaler_*.json"):
            match = re.match(r"scaler_(.+)\.json", file.name)
            if match:
                ticker = match.group(1).upper()
                scaler_tickers.add(ticker)

    return {
        "data": data_tickers,
        "scalers": scaler_tickers,
        "available": data_tickers,  # Use data files as source of truth
    }


class MultiStockDataProcessor:
    """Process data for multiple stocks"""

    def __init__(self, tickers: List[str], data_dir: Path, scalers_dir: Path):
        self.tickers = [t.upper() for t in tickers]
        self.data_dir = Path(data_dir)
        self.scalers_dir = Path(scalers_dir)
        self.scaler_params = {}
        self.feature_names = []

    def load_and_process_ticker(
        self, ticker: str, seq_len: int = 60, horizon: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process data for a single ticker"""
        # Load data
        ticker_file = self.data_dir / f"{ticker}.parquet"

        if not ticker_file.exists():
            # Try lowercase
            ticker_file_lower = self.data_dir / f"{ticker.lower()}.parquet"
            if ticker_file_lower.exists():
                ticker_file = ticker_file_lower
            else:
                logging.warning(f"No parquet file found for ticker {ticker}, skipping")
                return None, None

        df = pd.read_parquet(ticker_file)
        df = df.sort_index()

        # Engineer features
        features = self.engineer_features(df, ticker)

        # Normalize
        normalized = self.normalize_features(features, ticker)

        # Create sequences
        sequences, targets = self.create_sequences(normalized, seq_len, horizon)

        logging.info(f"Processed {ticker}: {len(sequences)} sequences")

        return sequences, targets

    def engineer_features(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """Engineer features from raw OHLCV data"""
        features = []

        # Basic OHLCV features
        features.append(df["Open"].values)
        features.append(df["High"].values)
        features.append(df["Low"].values)
        features.append(df["Close"].values)
        features.append(df["Volume"].values)

        # Technical indicators
        features.append(df["Close"].pct_change().fillna(0).values)
        features.append(df["Close"].rolling(5).mean().fillna(df["Close"]).values)
        features.append(df["Close"].rolling(20).mean().fillna(df["Close"]).values)

        # Volatility
        returns = df["Close"].pct_change()
        features.append(returns.rolling(20).std().fillna(0).values)

        # Volume MA
        features.append(df["Volume"].rolling(5).mean().fillna(df["Volume"]).values)

        if ticker == self.tickers[0]:  # Store feature names once
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

        return np.stack(features, axis=1)

    def normalize_features(self, features: np.ndarray, ticker: str) -> np.ndarray:
        """Normalize features per ticker"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8

        # Store scaler params per ticker
        self.scaler_params[ticker] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_names": self.feature_names,
            "ticker": ticker,
        }

        normalized = (features - mean) / std
        return normalized

    def save_scalers(self):
        """Save all scaler parameters"""
        self.scalers_dir.mkdir(parents=True, exist_ok=True)

        for ticker, params in self.scaler_params.items():
            scaler_path = self.scalers_dir / f"scaler_{ticker}.json"
            with open(scaler_path, "w") as f:
                json.dump(params, f, indent=2)
            logging.info(f"Saved scaler for {ticker}")

    def create_sequences(
        self, data: np.ndarray, seq_len: int, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        sequences = []
        targets = []

        for i in range(seq_len, len(data) - horizon):
            sequences.append(data[i - seq_len : i])
            targets.append(data[i : i + horizon, 3])  # Close price

        if len(sequences) == 0:
            return None, None

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return sequences, targets


def setup_logging():
    """Setup logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_multi_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_file


def main():
    """Main training pipeline for multiple stocks"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Stock Training")
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Stock tickers to train on (default: all available)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon")
    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension of transformer"
    )
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")

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

    args = parser.parse_args()

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
    log_file = setup_logging()

    # Log discovery information
    logging.info("=" * 60)
    logging.info("MULTI-STOCK TICKER DISCOVERY")
    logging.info("=" * 60)
    logging.info(f"Current working directory: {Path.cwd()}")
    logging.info(f"Data directory: {data_dir.resolve()}")
    logging.info(f"Scalers directory: {scalers_dir.resolve()}")

    # Discover available tickers
    ticker_info = discover_all_tickers(data_dir, scalers_dir)

    logging.info(f"\nDiscovered tickers from data: {sorted(ticker_info['data'])}")
    logging.info(f"Count: {len(ticker_info['data'])}")
    logging.info(f"\nDiscovered tickers from scalers: {sorted(ticker_info['scalers'])}")
    logging.info(f"Count: {len(ticker_info['scalers'])}")
    logging.info(f"\nAvailable tickers: {sorted(ticker_info['available'])}")
    logging.info("=" * 60)

    # Determine which tickers to use
    if args.tickers:
        # Use specified tickers that are available
        tickers = [t.upper() for t in args.tickers if t.upper() in ticker_info["available"]]
        if not tickers:
            raise ValueError(
                f"None of the specified tickers {args.tickers} are available. "
                f"Available: {sorted(ticker_info['available'])}"
            )
    else:
        # Use all available tickers
        tickers = sorted(ticker_info["available"])

    if not tickers:
        raise ValueError(f"No tickers found in {data_dir}. Please check your data directory.")

    logging.info(f"Training on {len(tickers)} tickers: {tickers}")

    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project="timeseries-transformer",
            name=f"multi_stock_{len(tickers)}tickers_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={**vars(args), "tickers": tickers},
            tags=["multi_stock"] + [t.lower() for t in tickers],
        )

    try:
        # Initialize data processor
        processor = MultiStockDataProcessor(tickers, data_dir=data_dir, scalers_dir=scalers_dir)

        # Process all tickers
        all_sequences = []
        all_targets = []
        ticker_ids = []

        for idx, ticker in enumerate(tickers):
            sequences, targets = processor.load_and_process_ticker(
                ticker, seq_len=args.seq_len, horizon=args.horizon
            )

            if sequences is not None and targets is not None:
                all_sequences.append(sequences)
                all_targets.append(targets)
                # Create ticker ID tensor for this ticker's sequences
                ticker_ids.extend([idx] * len(sequences))

        # Concatenate all data
        all_sequences = np.concatenate(all_sequences, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        ticker_ids = np.array(ticker_ids, dtype=np.int64)

        logging.info(f"Total sequences: {len(all_sequences)}")
        logging.info(f"Sequence shape: {all_sequences.shape}")
        logging.info(f"Target shape: {all_targets.shape}")

        # Verify no NaN/Inf
        assert not np.any(np.isnan(all_sequences)), "NaN in sequences"
        assert not np.any(np.isinf(all_sequences)), "Inf in sequences"
        assert not np.any(np.isnan(all_targets)), "NaN in targets"
        assert not np.any(np.isinf(all_targets)), "Inf in targets"

        # Save scalers
        processor.save_scalers()

        # Convert to tensors
        sequences_tensor = torch.FloatTensor(all_sequences)
        targets_tensor = torch.FloatTensor(all_targets)
        ticker_ids_tensor = torch.LongTensor(ticker_ids)

        # Create dataset
        dataset = TensorDataset(sequences_tensor, targets_tensor, ticker_ids_tensor)

        # Split into train/val
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size

        # Use random split
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        logging.info(f"Train samples: {train_size}, Val samples: {val_size}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Get input dimension from first ticker's features
        input_dim = len(processor.feature_names)

        model = TimeSeriesTransformer(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=args.horizon,
            dropout=args.dropout,
        ).to(device)

        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            logging.info(f"\nEpoch {epoch}/{args.epochs}")

            # Training
            model.train()
            train_loss = 0
            num_batches = 0

            for batch_idx, (sequences, targets, _) in enumerate(train_loader):
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Forward pass
                predictions = model(sequences)
                loss = criterion(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / num_batches

            # Validation
            model.eval()
            val_loss = 0
            val_mse = 0
            num_val_batches = 0

            with torch.no_grad():
                for sequences, targets, _ in val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device)

                    predictions = model(sequences)
                    loss = criterion(predictions, targets)

                    val_loss += loss.item()
                    val_mse += torch.mean((predictions - targets) ** 2).item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            val_rmse = np.sqrt(val_mse / num_val_batches)

            logging.info(f"Train Loss: {avg_train_loss:.4f}")
            logging.info(f"Val Loss: {avg_val_loss:.4f}")
            logging.info(f"Val RMSE: {val_rmse:.4f}")

            # Log to W&B
            if args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_rmse": val_rmse,
                    }
                )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                # Save checkpoint
                save_dir = Path("models")
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"model_multi_{len(tickers)}stocks_best.pt"

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_rmse": val_rmse,
                    "tickers": tickers,
                    "timestamp": datetime.now().isoformat(),
                    "config": vars(args),
                }

                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved best model to {checkpoint_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch} epochs")
                    break

        # Final summary
        logging.info("\n" + "=" * 60)
        logging.info("TRAINING COMPLETE")
        logging.info(f"Trained on {len(tickers)} tickers: {tickers}")
        logging.info(f"Best Val Loss: {best_val_loss:.4f}")
        logging.info(f"Model saved to: models/model_multi_{len(tickers)}stocks_best.pt")
        logging.info(f"Scalers saved to: {scalers_dir}")
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
