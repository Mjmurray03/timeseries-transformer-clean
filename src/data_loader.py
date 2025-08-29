"""
Data Loading and Preprocessing for Time-Series Transformer

Clean implementation focused on the working data pipeline.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
from pathlib import Path


class StockDataset(Dataset):
    """Simple dataset for stock sequences and targets."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences (N, seq_len, features)
            targets: Target values (N, horizon)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
        assert len(self.sequences) == len(self.targets)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class FeatureEngineer:
    """Simple feature engineering with essential technical indicators."""
    
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with engineered features
        """
        result = data.copy()
        
        # Basic price features
        result['Returns'] = result['Close'].pct_change()
        result['HL_Ratio'] = (result['High'] - result['Low']) / result['Close']
        result['OC_Ratio'] = (result['Close'] - result['Open']) / result['Open']
        
        # Simple moving averages
        result['SMA_5'] = result['Close'].rolling(5).mean()
        result['SMA_20'] = result['Close'].rolling(20).mean()
        
        # RSI
        result['RSI'] = self._calculate_rsi(result['Close'])
        
        # Volume features
        result['Volume_MA'] = result['Volume'].rolling(20).mean()
        result['Volume_Ratio'] = result['Volume'] / result['Volume_MA']
        
        # Store feature names (excluding original OHLCV)
        self.feature_names = [col for col in result.columns 
                             if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date', 'ticker', 'Ticker']]
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI for initial values


def load_stock_data(data_path: str = "data/raw", ticker: str = "AAPL") -> pd.DataFrame:
    """
    Load stock data from parquet files.
    
    Args:
        data_path: Path to data directory
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with stock data
    """
    data_dir = Path(data_path)
    
    # Look for ticker-specific data
    ticker_files = list(data_dir.glob(f"**/*{ticker}*.parquet"))
    
    if not ticker_files:
        # Fallback to any parquet files
        ticker_files = list(data_dir.glob("**/*.parquet"))
    
    if not ticker_files:
        raise FileNotFoundError(f"No data files found for {ticker} in {data_path}")
    
    # Load the most recent file
    data_file = ticker_files[0]
    print(f"Loading data from: {data_file}")
    
    df = pd.read_parquet(data_file)
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} rows of data for {ticker}")
    return df


def create_sequences(
    data: np.ndarray, 
    seq_length: int = 60, 
    forecast_horizon: int = 3,
    target_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Feature array (timesteps, features)
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to predict
        target_col: Column index for target values
        
    Returns:
        Tuple of (sequences, targets)
    """
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data) - forecast_horizon):
        # Input sequence
        seq = data[i-seq_length:i]
        sequences.append(seq)
        
        # Target values (future prices)
        target = data[i:i+forecast_horizon, target_col]
        targets.append(target)
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_data(
    ticker: str = "AAPL",
    data_path: str = "data/raw",
    seq_length: int = 60,
    forecast_horizon: int = 3,
    test_size: float = 0.2
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Complete data preparation pipeline.
    
    Args:
        ticker: Stock ticker
        data_path: Path to data files
        seq_length: Input sequence length
        forecast_horizon: Prediction horizon
        test_size: Fraction of data for testing
        
    Returns:
        Tuple of (train_loader, test_loader, num_features)
    """
    # Load raw data
    df = load_stock_data(data_path, ticker)
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df)
    
    # Drop rows with NaN values
    df_clean = df_features.dropna()
    print(f"After cleaning: {len(df_clean)} rows")
    
    # Get feature columns
    feature_cols = feature_engineer.feature_names
    print(f"Using {len(feature_cols)} features")
    
    # Convert to numpy array
    feature_data = df_clean[feature_cols].values.astype(np.float32)
    
    # Create sequences
    sequences, targets = create_sequences(
        feature_data, seq_length, forecast_horizon, target_col=0  # Returns column
    )
    print(f"Created {len(sequences)} sequences")
    
    # Train/test split
    split_idx = int(len(sequences) * (1 - test_size))
    
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    test_sequences = sequences[split_idx:]
    test_targets = targets[split_idx:]
    
    # Create datasets
    train_dataset = StockDataset(train_sequences, train_targets)
    test_dataset = StockDataset(test_sequences, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    return train_loader, test_loader, len(feature_cols)