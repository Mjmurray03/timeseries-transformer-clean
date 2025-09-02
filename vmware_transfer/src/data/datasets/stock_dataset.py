"""Stock dataset implementation for time-series transformer training."""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StockSequenceDataset(Dataset):
    """Dataset for stock price sequences with targets."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        features: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize stock sequence dataset.
        
        Args:
            sequences: Input sequences (N, window_size, num_features)
            targets: Target values (N, horizon)
            features: List of feature names
            transform: Optional transform function
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.features = features or []
        self.transform = transform
        
        # Validate shapes
        assert len(self.sequences) == len(self.targets), \
            f"Sequences and targets must have same length: {len(self.sequences)} vs {len(self.targets)}"
        
        logger.info(f"Created dataset with {len(self)} samples")
        logger.info(f"Sequence shape: {self.sequences.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'inputs' and 'targets' tensors
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Apply transform if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        return {
            'inputs': sequence,
            'targets': target,
            'index': torch.tensor(idx, dtype=torch.long)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.features
    
    def get_stats(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'sequence_length': self.sequences.shape[1],
            'num_features': self.sequences.shape[2],
            'target_horizon': self.targets.shape[1],
            'mean_target': float(self.targets.mean()),
            'std_target': float(self.targets.std()),
            'min_target': float(self.targets.min()),
            'max_target': float(self.targets.max())
        }


class MultiStockDataset(Dataset):
    """Dataset for multiple stocks with metadata."""
    
    def __init__(
        self,
        stock_data: Dict[str, Dict[str, np.ndarray]],
        features: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize multi-stock dataset.
        
        Args:
            stock_data: Dict of {ticker: {'sequences': array, 'targets': array}}
            features: List of feature names
            transform: Optional transform function
        """
        self.stock_data = stock_data
        self.features = features or []
        self.transform = transform
        
        # Create index mapping
        self.index_map = []
        for ticker, data in stock_data.items():
            num_samples = len(data['sequences'])
            for i in range(num_samples):
                self.index_map.append((ticker, i))
        
        logger.info(f"Created multi-stock dataset with {len(self)} samples from {len(stock_data)} stocks")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'inputs', 'targets', 'ticker', and 'index'
        """
        ticker, sample_idx = self.index_map[idx]
        data = self.stock_data[ticker]
        
        sequence = torch.FloatTensor(data['sequences'][sample_idx])
        target = torch.FloatTensor(data['targets'][sample_idx])
        
        # Apply transform if provided
        if self.transform:
            sequence = self.transform(sequence)
        
        return {
            'inputs': sequence,
            'targets': target,
            'ticker': ticker,
            'index': torch.tensor(idx, dtype=torch.long)
        }
    
    def get_tickers(self) -> List[str]:
        """Get list of tickers."""
        return list(self.stock_data.keys())
    
    def get_ticker_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics per ticker."""
        stats = {}
        for ticker, data in self.stock_data.items():
            targets = data['targets']
            stats[ticker] = {
                'num_samples': len(targets),
                'mean_target': float(np.mean(targets)),
                'std_target': float(np.std(targets)),
                'min_target': float(np.min(targets)),
                'max_target': float(np.max(targets))
            }
        return stats


class DataAugmentation:
    """Data augmentation transforms for time series."""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        dropout_prob: float = 0.1,
        time_shift_prob: float = 0.2,
        magnitude_warp_prob: float = 0.2
    ):
        """
        Initialize data augmentation.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            dropout_prob: Probability of feature dropout
            time_shift_prob: Probability of time shifting
            magnitude_warp_prob: Probability of magnitude warping
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.time_shift_prob = time_shift_prob
        self.magnitude_warp_prob = magnitude_warp_prob
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to sequence.
        
        Args:
            sequence: Input sequence (seq_len, num_features)
            
        Returns:
            Augmented sequence
        """
        sequence = sequence.clone()
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(sequence) * self.noise_std
            sequence += noise
        
        # Feature dropout
        if self.dropout_prob > 0 and torch.rand(1) < self.dropout_prob:
            # Randomly zero out some features
            mask = torch.rand(sequence.shape[1]) > self.dropout_prob
            sequence[:, ~mask] = 0
        
        # Time shifting (circular shift)
        if self.time_shift_prob > 0 and torch.rand(1) < self.time_shift_prob:
            shift = torch.randint(-5, 6, (1,)).item()
            sequence = torch.roll(sequence, shift, dims=0)
        
        # Magnitude warping
        if self.magnitude_warp_prob > 0 and torch.rand(1) < self.magnitude_warp_prob:
            warp_factor = torch.normal(1.0, 0.1, (sequence.shape[1],))
            sequence *= warp_factor.unsqueeze(0)
        
        return sequence


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle_train: Whether to shuffle training data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    logger.info(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_loader)} batches, {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = False,
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Split sequences into train/val/test sets.
    
    Args:
        sequences: Input sequences
        targets: Target values
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        
    Returns:
        Tuple of ((train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets))
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(sequences)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
        sequences = sequences[indices]
        targets = targets[indices]
    
    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    # Split data
    train_seq = sequences[:train_end]
    train_targets = targets[:train_end]
    
    val_seq = sequences[train_end:val_end]
    val_targets = targets[train_end:val_end]
    
    test_seq = sequences[val_end:]
    test_targets = targets[val_end:]
    
    logger.info(f"Split {n_samples} samples into:")
    logger.info(f"  Train: {len(train_seq)} ({len(train_seq)/n_samples:.1%})")
    logger.info(f"  Val: {len(val_seq)} ({len(val_seq)/n_samples:.1%})")
    logger.info(f"  Test: {len(test_seq)} ({len(test_seq)/n_samples:.1%})")
    
    return (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets)


class SequenceCollator:
    """Custom collate function for variable-length sequences."""
    
    def __init__(self, pad_value: float = 0.0):
        """Initialize collator with padding value."""
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched tensors
        """
        # Extract components
        inputs = [item['inputs'] for item in batch]
        targets = [item['targets'] for item in batch]
        indices = [item['index'] for item in batch]
        
        # Stack tensors (assuming same shape)
        batched = {
            'inputs': torch.stack(inputs),
            'targets': torch.stack(targets),
            'indices': torch.stack(indices)
        }
        
        # Add ticker information if present
        if 'ticker' in batch[0]:
            batched['tickers'] = [item['ticker'] for item in batch]
        
        return batched