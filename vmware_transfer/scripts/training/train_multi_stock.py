#!/usr/bin/env python3
"""
# COMPONENT: Multi-Stock Unified Training Script
# PURPOSE: Train single transformer model on multiple stocks simultaneously
# INPUTS: Multiple ticker data from data/raw/, ticker embeddings for differentiation
# OUTPUTS: Unified model with ticker awareness, per-ticker scalers, comprehensive metrics
# VERIFICATION: Per-ticker validation, stratified sampling, weighted loss by volatility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import json
import wandb
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


class MultiStockDataset(Dataset):
    """
    # COMPONENT: Multi-Stock Dataset
    # PURPOSE: Handle multiple stocks with ticker identification
    # INPUTS: Sequences from multiple stocks, ticker IDs
    # OUTPUTS: Batches with (sequence, target, ticker_id)
    # VERIFICATION: Maintains ticker balance, handles missing data
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        ticker_ids: np.ndarray,
        ticker_names: List[str]
    ):
        assert len(sequences) == len(targets) == len(ticker_ids), "Data length mismatch"
        
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.ticker_ids = torch.LongTensor(ticker_ids)
        self.ticker_names = ticker_names
        self.n_tickers = len(ticker_names)
        
        # Calculate ticker distribution
        self.ticker_counts = {}
        for ticker_idx in range(self.n_tickers):
            count = (self.ticker_ids == ticker_idx).sum().item()
            self.ticker_counts[ticker_names[ticker_idx]] = count
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.targets[idx],
            self.ticker_ids[idx]
        )
    
    def get_ticker_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of tickers in dataset"""
        total = len(self)
        return {
            ticker: count / total 
            for ticker, count in self.ticker_counts.items()
        }


class MultiStockProcessor:
    """
    # COMPONENT: Multi-Stock Data Processor
    # PURPOSE: Load and process multiple stocks with per-ticker normalization
    # INPUTS: List of tickers, data directory
    # OUTPUTS: Concatenated sequences with ticker IDs and per-ticker scalers
    # VERIFICATION: Independent scaling per ticker, handles different date ranges
    """
    
    def __init__(self, tickers: List[str], data_dir: Path = Path("data/raw")):
        self.tickers = [t.upper() for t in tickers]
        self.data_dir = data_dir
        self.scalers = {}
        self.feature_names = []
        self.ticker_to_id = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.volatilities = {}
        
    def load_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """Load data for all specified tickers"""
        ticker_data = {}
        
        for ticker in self.tickers:
            ticker_dir = self.data_dir / ticker
            if not ticker_dir.exists():
                logging.warning(f"Ticker directory not found: {ticker}")
                continue
            
            parquet_files = list(ticker_dir.glob("*.parquet"))
            if not parquet_files:
                logging.warning(f"No parquet files for {ticker}")
                continue
            
            # Load most recent file
            parquet_files.sort()
            df = pd.read_parquet(parquet_files[-1])
            
            # Validate required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required):
                ticker_data[ticker] = df
                logging.info(f"Loaded {ticker}: {len(df)} rows")
            else:
                logging.warning(f"Missing columns for {ticker}")
        
        return ticker_data
    
    def engineer_features(self, df: pd.DataFrame, ticker: str) -> np.ndarray:
        """
        Engineer features for a single ticker
        Consistent feature set across all tickers
        """
        features = []
        feature_names = []
        
        # Core price features
        for col in ['Open', 'High', 'Low', 'Close']:
            features.append(df[col].values)
            feature_names.append(col)
        
        # Log volume
        volume = df['Volume'].values
        log_volume = np.log1p(volume)
        features.append(log_volume)
        feature_names.append('LogVolume')
        
        # Price-based features
        close = df['Close'].values
        
        # Returns
        returns = np.zeros_like(close)
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
        features.append(returns)
        feature_names.append('Returns')
        
        # Calculate and store volatility for this ticker
        volatility = np.std(returns[1:])
        self.volatilities[ticker] = volatility
        
        # Moving averages
        for window in [5, 10, 20]:
            ma = pd.Series(close).rolling(window=window, min_periods=1).mean().values
            features.append(ma)
            feature_names.append(f'MA_{window}')
        
        # RSI
        rsi = self.calculate_rsi(close, period=14)
        features.append(rsi)
        feature_names.append('RSI')
        
        # Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(close, window=20)
        features.append(bb_upper)
        features.append(bb_lower)
        feature_names.append('BB_Upper')
        feature_names.append('BB_Lower')
        
        # Store feature names once
        if not self.feature_names:
            self.feature_names = feature_names
        
        # Stack and validate
        feature_array = np.column_stack(features).astype(np.float32)
        
        # Remove NaN/Inf rows
        valid_mask = ~(np.isnan(feature_array).any(axis=1) | np.isinf(feature_array).any(axis=1))
        feature_array = feature_array[valid_mask]
        
        logging.info(f"{ticker}: {feature_array.shape[0]} valid rows after feature engineering")
        
        return feature_array
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
        avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self, 
        prices: np.ndarray, 
        window: int = 20, 
        num_std: float = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        ma = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
        std = pd.Series(prices).rolling(window=window, min_periods=1).std().fillna(0).values
        
        upper_band = ma + (num_std * std)
        lower_band = ma - (num_std * std)
        
        return upper_band, lower_band
    
    def normalize_ticker_features(
        self, 
        features: np.ndarray, 
        ticker: str
    ) -> np.ndarray:
        """
        Normalize features independently for each ticker
        Critical for handling different price scales
        """
        normalized = np.zeros_like(features)
        scaler_params = {}
        
        for i, feature_name in enumerate(self.feature_names):
            col = features[:, i]
            
            # Calculate statistics
            mean = float(np.mean(col))
            std = float(np.std(col))
            
            # Prevent division by zero
            if std < 1e-8:
                std = 1.0
            
            # Normalize
            normalized[:, i] = (col - mean) / std
            
            # Store parameters
            scaler_params[feature_name] = {
                'mean': mean,
                'std': std,
                'min': float(np.min(col)),
                'max': float(np.max(col))
            }
        
        # Store ticker-specific scaler
        self.scalers[ticker] = scaler_params
        
        # Validate
        assert not np.any(np.isnan(normalized)), f"NaN after normalization for {ticker}"
        assert not np.any(np.isinf(normalized)), f"Inf after normalization for {ticker}"
        
        return normalized
    
    def create_unified_sequences(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        seq_len: int = 60,
        horizon: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from all tickers with ticker identification
        Returns: (sequences, targets, ticker_ids)
        """
        all_sequences = []
        all_targets = []
        all_ticker_ids = []
        
        for ticker, df in ticker_data.items():
            logging.info(f"Processing {ticker}...")
            
            # Engineer features
            features = self.engineer_features(df, ticker)
            
            # Normalize per ticker
            normalized = self.normalize_ticker_features(features, ticker)
            
            # Get close price index for targets
            close_idx = self.feature_names.index('Close')
            
            # Create sequences for this ticker
            for i in range(seq_len, len(normalized) - horizon):
                # Input sequence
                seq = normalized[i-seq_len:i]
                
                # Target: percentage returns
                current_close = features[i-1, close_idx]  # Use original scale
                future_closes = features[i:i+horizon, close_idx]
                pct_returns = (future_closes - current_close) / (current_close + 1e-8)
                
                all_sequences.append(seq)
                all_targets.append(pct_returns)
                all_ticker_ids.append(self.ticker_to_id[ticker])
            
            logging.info(f"{ticker}: Created {len(all_sequences) - len(all_ticker_ids) + self.ticker_to_id[ticker]} sequences")
        
        # Convert to arrays
        sequences = np.array(all_sequences, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)
        ticker_ids = np.array(all_ticker_ids, dtype=np.int64)
        
        # Final validation
        assert sequences.shape[0] == targets.shape[0] == ticker_ids.shape[0]
        assert sequences.shape[1] == seq_len
        assert sequences.shape[2] == len(self.feature_names)
        assert targets.shape[1] == horizon
        
        logging.info(f"Total sequences: {len(sequences)}")
        logging.info(f"Sequences shape: {sequences.shape}")
        logging.info(f"Targets shape: {targets.shape}")
        logging.info(f"Unique tickers: {len(np.unique(ticker_ids))}")
        
        return sequences, targets, ticker_ids
    
    def save_scalers(self, save_dir: Path):
        """Save all ticker scalers in a unified file"""
        save_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = save_dir / "scaler_multi_stock.json"
        
        scaler_data = {
            'tickers': self.tickers,
            'ticker_to_id': self.ticker_to_id,
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'volatilities': self.volatilities,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        logging.info(f"Saved multi-stock scaler to {scaler_path}")


class MultiStockTransformer(TimeSeriesTransformer):
    """
    # COMPONENT: Multi-Stock Transformer Model
    # PURPOSE: Extend base transformer with ticker embeddings
    # INPUTS: Sequences with ticker IDs
    # OUTPUTS: Predictions with ticker-aware representations
    # VERIFICATION: Embedding dimension matches, proper concatenation
    """
    
    def __init__(
        self,
        n_tickers: int,
        embedding_dim: int = 16,
        input_dim: int = 10,
        hidden_dim: int = 256,
        **kwargs
    ):
        # Adjust input dimension to account for ticker embedding
        adjusted_input_dim = input_dim + embedding_dim
        
        # Initialize base transformer with adjusted input
        super().__init__(
            input_dim=adjusted_input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
        
        # Store original dimensions
        self.original_input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_tickers = n_tickers
        
        # Ticker embedding layer
        self.ticker_embedding = nn.Embedding(
            num_embeddings=n_tickers,
            embedding_dim=embedding_dim
        )
        
        # Initialize embedding weights
        nn.init.normal_(self.ticker_embedding.weight, mean=0, std=0.02)
        
        logging.info(f"MultiStockTransformer initialized with {n_tickers} tickers, embedding_dim={embedding_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        ticker_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with ticker embedding concatenation
        
        Args:
            x: Input sequences (batch_size, seq_len, original_input_dim)
            ticker_ids: Ticker IDs (batch_size,)
            mask: Optional attention mask
        """
        batch_size, seq_len, features = x.shape
        
        # Validate input dimensions
        assert features == self.original_input_dim, f"Expected {self.original_input_dim} features, got {features}"
        
        # Get ticker embeddings
        ticker_embeds = self.ticker_embedding(ticker_ids)  # (batch_size, embedding_dim)
        
        # Expand to match sequence length
        ticker_embeds = ticker_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
        
        # Concatenate ticker embedding to each timestep
        x_with_ticker = torch.cat([x, ticker_embeds], dim=-1)  # (batch_size, seq_len, input_dim + embedding_dim)
        
        # Call parent forward method
        return super().forward(x_with_ticker, mask=mask, **kwargs)


class WeightedMultiStockTrainer:
    """
    # COMPONENT: Weighted Multi-Stock Trainer
    # PURPOSE: Train with per-ticker metrics and volatility weighting
    # INPUTS: Multi-stock model and dataset
    # OUTPUTS: Trained model with per-ticker performance tracking
    # VERIFICATION: Stratified validation, weighted loss, per-ticker metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ticker_names: List[str],
        volatilities: Dict[str, float],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.ticker_names = ticker_names
        self.n_tickers = len(ticker_names)
        self.volatilities = volatilities
        
        # Calculate volatility weights (inverse volatility weighting)
        vol_values = np.array([volatilities.get(t, 1.0) for t in ticker_names])
        self.vol_weights = 1.0 / (vol_values + 1e-8)
        self.vol_weights = self.vol_weights / self.vol_weights.sum()
        self.vol_weights = torch.FloatTensor(self.vol_weights).to(device)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Per-ticker metrics tracking
        self.ticker_metrics = defaultdict(list)
        
    def compute_weighted_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ticker_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss weighted by ticker volatility
        Low volatility stocks get higher weight
        """
        base_loss = nn.MSELoss(reduction='none')(predictions, targets)
        base_loss = base_loss.mean(dim=1)  # Average over horizon
        
        # Apply volatility weights
        ticker_weights = self.vol_weights[ticker_ids]
        weighted_loss = base_loss * ticker_weights
        
        return weighted_loss.mean()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train with per-ticker loss tracking"""
        self.model.train()
        
        ticker_losses = defaultdict(list)
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, targets, ticker_ids) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            ticker_ids = ticker_ids.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences, ticker_ids)
            
            # Ensure shape match
            if predictions.shape != targets.shape:
                predictions = predictions[:, :targets.shape[1]]
            
            # Compute weighted loss
            loss = self.compute_weighted_loss(predictions, targets, ticker_ids)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss at batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track per-ticker losses
            for ticker_idx in range(self.n_tickers):
                ticker_mask = (ticker_ids == ticker_idx)
                if ticker_mask.any():
                    ticker_loss = nn.MSELoss()(
                        predictions[ticker_mask],
                        targets[ticker_mask]
                    ).item()
                    ticker_losses[self.ticker_names[ticker_idx]].append(ticker_loss)
            
            total_loss += loss.item()
            num_batches += 1
        
        # Aggregate metrics
        metrics = {
            'train_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }
        
        for ticker in self.ticker_names:
            if ticker in ticker_losses and ticker_losses[ticker]:
                metrics[f'train_loss_{ticker}'] = np.mean(ticker_losses[ticker])
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate with per-ticker metrics"""
        self.model.eval()
        
        ticker_predictions = defaultdict(list)
        ticker_targets = defaultdict(list)
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, targets, ticker_ids in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                ticker_ids = ticker_ids.to(self.device)
                
                # Forward pass
                predictions = self.model(sequences, ticker_ids)
                
                # Ensure shape match
                if predictions.shape != targets.shape:
                    predictions = predictions[:, :targets.shape[1]]
                
                # Compute loss
                loss = self.compute_weighted_loss(predictions, targets, ticker_ids)
                total_loss += loss.item()
                num_batches += 1
                
                # Store per-ticker predictions
                for ticker_idx in range(self.n_tickers):
                    ticker_mask = (ticker_ids == ticker_idx)
                    if ticker_mask.any():
                        ticker_name = self.ticker_names[ticker_idx]
                        ticker_predictions[ticker_name].append(predictions[ticker_mask].cpu())
                        ticker_targets[ticker_name].append(targets[ticker_mask].cpu())
        
        # Calculate metrics
        metrics = {
            'val_loss': total_loss / num_batches if num_batches > 0 else float('inf')
        }
        
        # Per-ticker metrics
        for ticker in self.ticker_names:
            if ticker in ticker_predictions:
                preds = torch.cat(ticker_predictions[ticker], dim=0)
                targs = torch.cat(ticker_targets[ticker], dim=0)
                
                mse = nn.MSELoss()(preds, targs).item()
                rmse = np.sqrt(mse)
                mae = nn.L1Loss()(preds, targs).item()
                
                # Direction accuracy
                pred_dir = (preds > 0).float()
                true_dir = (targs > 0).float()
                dir_acc = (pred_dir == true_dir).float().mean().item()
                
                metrics[f'val_rmse_{ticker}'] = rmse
                metrics[f'val_mae_{ticker}'] = mae
                metrics[f'val_dir_acc_{ticker}'] = dir_acc
        
        # Aggregate metrics
        all_rmse = [v for k, v in metrics.items() if 'rmse' in k]
        if all_rmse:
            metrics['val_rmse_mean'] = np.mean(all_rmse)
        
        return metrics


def create_stratified_split(
    dataset: MultiStockDataset,
    val_ratio: float = 0.2
) -> Tuple[Dataset, Dataset]:
    """
    Create train/val split with stratification by ticker
    Ensures each ticker is represented proportionally
    """
    indices_by_ticker = defaultdict(list)
    
    # Group indices by ticker
    for idx in range(len(dataset)):
        _, _, ticker_id = dataset[idx]
        indices_by_ticker[ticker_id.item()].append(idx)
    
    train_indices = []
    val_indices = []
    
    # Split each ticker's indices
    for ticker_id, indices in indices_by_ticker.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * (1 - val_ratio))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def main():
    """
    # COMPONENT: Main Multi-Stock Training Pipeline
    # PURPOSE: Orchestrate unified multi-stock model training
    # INPUTS: Multiple ticker data, training configuration
    # OUTPUTS: Unified model, per-ticker scalers, comprehensive metrics
    # VERIFICATION: All tickers processed, stratified validation, W&B logging
    """
    
    parser = argparse.ArgumentParser(description='Multi-Stock Unified Training')
    parser.add_argument('--tickers', type=str, nargs='+',
                        default=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA'],
                        help='List of tickers to train on')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=60,
                        help='Sequence length')
    parser.add_argument('--horizon', type=int, default=3,
                        help='Prediction horizon')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of transformer')
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help='Ticker embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--use-stratified', action='store_true', default=True,
                        help='Use stratified sampling for batches')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"multi_stock_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting Multi-Stock Training")
    logging.info(f"Tickers: {args.tickers}")
    logging.info(f"Configuration: {vars(args)}")
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(
            project="timeseries-transformer",
            name=f"multi_stock_{timestamp}",
            config=vars(args),
            tags=["multi_stock", "unified"] + args.tickers
        )
    
    try:
        # Initialize processor
        processor = MultiStockProcessor(args.tickers)
        
        # Load all ticker data
        logging.info("Loading ticker data...")
        ticker_data = processor.load_all_tickers()
        
        if not ticker_data:
            raise ValueError("No valid ticker data found")
        
        logging.info(f"Loaded data for {len(ticker_data)} tickers")
        
        # Create unified sequences
        sequences, targets, ticker_ids = processor.create_unified_sequences(
            ticker_data,
            seq_len=args.seq_len,
            horizon=args.horizon
        )
        
        # Create dataset
        dataset = MultiStockDataset(
            sequences=sequences,
            targets=targets,
            ticker_ids=ticker_ids,
            ticker_names=list(ticker_data.keys())
        )
        
        # Log distribution
        distribution = dataset.get_ticker_distribution()
        logging.info("Ticker distribution:")
        for ticker, pct in distribution.items():
            logging.info(f"  {ticker}: {pct:.2%}")
        
        # Create stratified split
        train_dataset, val_dataset = create_stratified_split(dataset, args.val_split)
        logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create data loaders
        if args.use_stratified:
            # Calculate sample weights for balanced sampling
            sample_weights = []
            for idx in range(len(train_dataset)):
                _, _, ticker_id = train_dataset[idx]
                # Inverse frequency weighting
                ticker_count = dataset.ticker_counts[dataset.ticker_names[ticker_id.item()]]
                weight = 1.0 / ticker_count
                sample_weights.append(weight)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True
            )
        else:
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
        
        # Initialize model
        model = MultiStockTransformer(
            n_tickers=len(ticker_data),
            embedding_dim=args.embedding_dim,
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
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model parameters: {total_params:,}")
        
        # Initialize trainer
        trainer = WeightedMultiStockTrainer(
            model=model,
            device=device,
            ticker_names=list(ticker_data.keys()),
            volatilities=processor.volatilities,
            learning_rate=args.learning_rate
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(args.epochs):
            logging.info(f"\n{'='*50}")
            logging.info(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Update scheduler
            trainer.scheduler.step()
            
            # Log metrics
            logging.info(f"Train Loss: {train_metrics['train_loss']:.6f}")
            logging.info(f"Val Loss: {val_metrics['val_loss']:.6f}")
            
            if 'val_rmse_mean' in val_metrics:
                logging.info(f"Val RMSE (mean): {val_metrics['val_rmse_mean']:.6f}")
            
            # Log per-ticker metrics
            logging.info("Per-ticker validation metrics:")
            for ticker in dataset.ticker_names:
                rmse_key = f'val_rmse_{ticker}'
                if rmse_key in val_metrics:
                    logging.info(f"  {ticker}: RMSE={val_metrics[rmse_key]:.4f}, "
                               f"MAE={val_metrics[f'val_mae_{ticker}']:.4f}, "
                               f"Dir={val_metrics[f'val_dir_acc_{ticker}']:.2%}")
            
            # Log to W&B
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **train_metrics,
                    **val_metrics
                })
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_metrics = val_metrics
                patience_counter = 0
                
                # Save checkpoint
                save_dir = Path("models")
                save_dir.mkdir(exist_ok=True)
                checkpoint_path = save_dir / "model_multi_stock_best.pt"
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'tickers': list(ticker_data.keys()),
                    'ticker_to_id': processor.ticker_to_id,
                    'timestamp': datetime.now().isoformat(),
                    'config': vars(args)
                }
                
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save scalers
        processor.save_scalers(Path("models/scalers"))
        
        # Final summary
        logging.info("\n" + "="*60)
        logging.info("TRAINING COMPLETE")
        logging.info("="*60)
        logging.info(f"Best Val Loss: {best_val_loss:.6f}")
        
        if 'val_rmse_mean' in best_metrics:
            logging.info(f"Best Mean RMSE: {best_metrics['val_rmse_mean']:.6f}")
        
        logging.info("\nBest per-ticker performance:")
        for ticker in dataset.ticker_names:
            rmse_key = f'val_rmse_{ticker}'
            if rmse_key in best_metrics:
                logging.info(f"  {ticker}: RMSE={best_metrics[rmse_key]:.4f}")
        
        logging.info(f"\nModel saved to: models/model_multi_stock_best.pt")
        logging.info(f"Scalers saved to: models/scalers/scaler_multi_stock.json")
        
        if args.use_wandb:
            wandb.summary.update(best_metrics)
            wandb.finish()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        if args.use_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()