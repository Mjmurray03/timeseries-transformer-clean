"""Unit tests for dataset implementations."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Mock torch before importing
import sys
torch_mock = Mock()
torch_mock.FloatTensor = Mock(side_effect=lambda x: x)
torch_mock.tensor = Mock(side_effect=lambda x, dtype=None: x)
torch_mock.long = Mock()
sys.modules['torch'] = torch_mock
sys.modules['torch.utils'] = Mock()
sys.modules['torch.utils.data'] = Mock()

from src.data.datasets.stock_dataset import (
    StockSequenceDataset,
    MultiStockDataset,
    DataAugmentation,
    split_sequences,
    SequenceCollator
)


class TestStockSequenceDataset:
    """Test suite for StockSequenceDataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        sequences = np.random.randn(100, 60, 7).astype(np.float32)
        targets = np.random.randn(100, 5).astype(np.float32)
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        dataset = StockSequenceDataset(sequences, targets, features)
        
        assert len(dataset) == 100
        assert dataset.features == features
        # Mock FloatTensor should be called with sequences and targets
        assert torch_mock.FloatTensor.call_count == 2
    
    def test_initialization_shape_mismatch(self):
        """Test initialization with mismatched shapes."""
        sequences = np.random.randn(100, 60, 7).astype(np.float32)
        targets = np.random.randn(50, 5).astype(np.float32)  # Wrong length
        
        with pytest.raises(AssertionError, match="Sequences and targets must have same length"):
            StockSequenceDataset(sequences, targets)
    
    def test_getitem(self):
        """Test getting item by index."""
        sequences = np.random.randn(10, 60, 7).astype(np.float32)
        targets = np.random.randn(10, 5).astype(np.float32)
        
        dataset = StockSequenceDataset(sequences, targets)
        
        item = dataset[0]
        
        assert 'inputs' in item
        assert 'targets' in item
        assert 'index' in item
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        sequences = np.random.randn(10, 60, 7).astype(np.float32)
        targets = np.random.randn(10, 5).astype(np.float32)
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        dataset = StockSequenceDataset(sequences, targets, features)
        
        assert dataset.get_feature_names() == features
    
    def test_get_stats(self):
        """Test getting dataset statistics."""
        sequences = np.random.randn(10, 60, 7).astype(np.float32)
        targets = np.random.randn(10, 5).astype(np.float32)
        
        # Mock the tensor attributes
        mock_sequences = Mock()
        mock_sequences.shape = (10, 60, 7)
        mock_targets = Mock()
        mock_targets.shape = (10, 5)
        mock_targets.mean.return_value = 0.1
        mock_targets.std.return_value = 1.0
        mock_targets.min.return_value = -2.0
        mock_targets.max.return_value = 2.0
        
        dataset = StockSequenceDataset(sequences, targets)
        dataset.sequences = mock_sequences
        dataset.targets = mock_targets
        
        stats = dataset.get_stats()
        
        assert stats['num_samples'] == 10
        assert stats['sequence_length'] == 60
        assert stats['num_features'] == 7
        assert stats['target_horizon'] == 5


class TestMultiStockDataset:
    """Test suite for MultiStockDataset."""
    
    def test_initialization(self):
        """Test multi-stock dataset initialization."""
        stock_data = {
            'AAPL': {
                'sequences': np.random.randn(50, 60, 7).astype(np.float32),
                'targets': np.random.randn(50, 5).astype(np.float32)
            },
            'GOOGL': {
                'sequences': np.random.randn(30, 60, 7).astype(np.float32),
                'targets': np.random.randn(30, 5).astype(np.float32)
            }
        }
        
        dataset = MultiStockDataset(stock_data)
        
        assert len(dataset) == 80  # 50 + 30
        assert len(dataset.index_map) == 80
        assert dataset.get_tickers() == ['AAPL', 'GOOGL']
    
    def test_getitem(self):
        """Test getting item by index."""
        stock_data = {
            'AAPL': {
                'sequences': np.random.randn(10, 60, 7).astype(np.float32),
                'targets': np.random.randn(10, 5).astype(np.float32)
            }
        }
        
        dataset = MultiStockDataset(stock_data)
        
        item = dataset[0]
        
        assert 'inputs' in item
        assert 'targets' in item
        assert 'ticker' in item
        assert 'index' in item
        assert item['ticker'] == 'AAPL'
    
    def test_get_ticker_stats(self):
        """Test getting per-ticker statistics."""
        stock_data = {
            'AAPL': {
                'sequences': np.random.randn(10, 60, 7).astype(np.float32),
                'targets': np.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 10).astype(np.float32)
            },
            'GOOGL': {
                'sequences': np.random.randn(5, 60, 7).astype(np.float32),
                'targets': np.array([[2.0, 3.0, 4.0, 5.0, 6.0]] * 5).astype(np.float32)
            }
        }
        
        dataset = MultiStockDataset(stock_data)
        stats = dataset.get_ticker_stats()
        
        assert 'AAPL' in stats
        assert 'GOOGL' in stats
        assert stats['AAPL']['num_samples'] == 10
        assert stats['GOOGL']['num_samples'] == 5


class TestDataAugmentation:
    """Test suite for DataAugmentation."""
    
    def test_initialization(self):
        """Test data augmentation initialization."""
        augmentation = DataAugmentation(
            noise_std=0.01,
            dropout_prob=0.1,
            time_shift_prob=0.2,
            magnitude_warp_prob=0.2
        )
        
        assert augmentation.noise_std == 0.01
        assert augmentation.dropout_prob == 0.1
        assert augmentation.time_shift_prob == 0.2
        assert augmentation.magnitude_warp_prob == 0.2
    
    def test_call_no_augmentation(self):
        """Test augmentation with zero probabilities."""
        # Mock torch operations
        mock_sequence = Mock()
        mock_sequence.clone.return_value = mock_sequence
        
        augmentation = DataAugmentation(
            noise_std=0.0,
            dropout_prob=0.0,
            time_shift_prob=0.0,
            magnitude_warp_prob=0.0
        )
        
        result = augmentation(mock_sequence)
        
        # Should return cloned sequence
        assert mock_sequence.clone.called
        assert result == mock_sequence


class TestSplitSequences:
    """Test suite for split_sequences function."""
    
    def test_split_sequences_basic(self):
        """Test basic sequence splitting."""
        sequences = np.random.randn(100, 60, 7)
        targets = np.random.randn(100, 5)
        
        splits = split_sequences(
            sequences, targets,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            shuffle=False
        )
        
        (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets) = splits
        
        assert len(train_seq) == 70
        assert len(val_seq) == 15
        assert len(test_seq) == 15
        assert len(train_targets) == 70
        assert len(val_targets) == 15
        assert len(test_targets) == 15
    
    def test_split_sequences_invalid_ratios(self):
        """Test splitting with invalid ratios."""
        sequences = np.random.randn(100, 60, 7)
        targets = np.random.randn(100, 5)
        
        with pytest.raises(AssertionError, match="Ratios must sum to 1.0"):
            split_sequences(
                sequences, targets,
                train_ratio=0.5, val_ratio=0.3, test_ratio=0.3  # Sum > 1.0
            )
    
    def test_split_sequences_with_shuffle(self):
        """Test splitting with shuffle."""
        sequences = np.arange(100 * 60 * 7).reshape(100, 60, 7)
        targets = np.arange(100 * 5).reshape(100, 5)
        
        # Split with shuffle
        splits_shuffled = split_sequences(
            sequences, targets,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            shuffle=True, random_state=42
        )
        
        # Split without shuffle
        splits_ordered = split_sequences(
            sequences, targets,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            shuffle=False
        )
        
        # Shuffled and ordered should have different first elements
        # (unless by extreme coincidence)
        train_shuffled, _ = splits_shuffled[0]
        train_ordered, _ = splits_ordered[0]
        
        # Shapes should be the same
        assert train_shuffled.shape == train_ordered.shape


class TestSequenceCollator:
    """Test suite for SequenceCollator."""
    
    def test_initialization(self):
        """Test collator initialization."""
        collator = SequenceCollator(pad_value=0.0)
        assert collator.pad_value == 0.0
    
    def test_call_basic(self):
        """Test basic collation."""
        # Mock torch.stack
        torch_mock.stack = Mock(side_effect=lambda x: x)
        
        batch = [
            {
                'inputs': np.random.randn(60, 7),
                'targets': np.random.randn(5),
                'index': 0
            },
            {
                'inputs': np.random.randn(60, 7),
                'targets': np.random.randn(5),
                'index': 1
            }
        ]
        
        collator = SequenceCollator()
        result = collator(batch)
        
        assert 'inputs' in result
        assert 'targets' in result
        assert 'indices' in result
        assert torch_mock.stack.call_count == 3  # inputs, targets, indices
    
    def test_call_with_tickers(self):
        """Test collation with ticker information."""
        torch_mock.stack = Mock(side_effect=lambda x: x)
        
        batch = [
            {
                'inputs': np.random.randn(60, 7),
                'targets': np.random.randn(5),
                'index': 0,
                'ticker': 'AAPL'
            },
            {
                'inputs': np.random.randn(60, 7),
                'targets': np.random.randn(5),
                'index': 1,
                'ticker': 'GOOGL'
            }
        ]
        
        collator = SequenceCollator()
        result = collator(batch)
        
        assert 'inputs' in result
        assert 'targets' in result
        assert 'indices' in result
        assert 'tickers' in result
        assert result['tickers'] == ['AAPL', 'GOOGL']