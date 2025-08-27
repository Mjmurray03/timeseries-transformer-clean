"""
Global test configuration and fixtures following testing-standards.md patterns.
"""
import os
import pytest
import torch
import numpy as np
import pandas as pd
import random
from unittest.mock import Mock, MagicMock
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Set test environment variables
os.environ["TESTING"] = "true"
os.environ["CACHE_ENABLED"] = "false"
os.environ["WANDB_MODE"] = "disabled"

# Import project modules
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.data.datasets.stock_dataset import StockSequenceDataset
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.training.trainer import TrainingOrchestrator as Trainer


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment for reproducible testing."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configure torch for testing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create test directories if needed
    test_dirs = ['data/test', 'models/test', 'cache/test']
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    cleanup_test_files()


def cleanup_test_files():
    """Remove temporary test files and directories."""
    import shutil
    test_dirs = ['data/test', 'models/test', 'cache/test']
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir, ignore_errors=True)


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast)")
    config.addinivalue_line("markers", "integration: Integration tests (slower)")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "smoke: Smoke tests for deployment")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Tests taking > 1 second")
    config.addinivalue_line("markers", "memory: Memory leak tests")
    config.addinivalue_line("markers", "load: Load testing")


# Model and configuration fixtures
@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        sequence_length=60,
        prediction_horizon=5,
        num_features=7,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        activation='gelu'
    )


@pytest.fixture
def training_config():
    """Create test training configuration."""
    return TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=2,
        patience=10,
        gradient_clip=1.0,
        weight_decay=1e-5
    )


@pytest.fixture
def test_model(model_config):
    """Create test transformer model instance."""
    model = TimeSeriesTransformer(model_config)
    model.eval()  # Set to evaluation mode for consistent testing
    return model


@pytest.fixture(scope="session")
def trained_model(model_config):
    """Session-scoped pre-trained model for expensive tests."""
    model = TimeSeriesTransformer(model_config)
    # Load or create minimal training state
    model.eval()
    return model


# Data fixtures
@pytest.fixture
def mock_stock_data():
    """Generate realistic mock stock data."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic price movements
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    high = prices * np.random.uniform(1.005, 1.02, n_days)
    low = prices * np.random.uniform(0.98, 0.995, n_days)
    volume = np.random.lognormal(15, 0.5, n_days).astype(int)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume,
        'Adj Close': prices
    }).set_index('Date')


@pytest.fixture
def mock_features_data(mock_stock_data):
    """Generate engineered features for testing."""
    data = mock_stock_data.copy()
    
    # Simple technical indicators
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['RSI'] = 50.0  # Simplified RSI
    data['MACD'] = data['Close'].ewm(12).mean() - data['Close'].ewm(26).mean()
    data['BB_upper'] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['BB_lower'] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
    data['Volume_MA'] = data['Volume'].rolling(10).mean()
    
    # Returns
    data['Returns'] = data['Close'].pct_change()
    
    # Fill NaNs
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    return data


@pytest.fixture
def test_sequences(mock_features_data):
    """Generate test sequences for model input."""
    feature_cols = ['Close', 'Volume', 'SMA_5', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
    data = mock_features_data[feature_cols].values
    
    sequences = []
    targets = []
    
    for i in range(60, len(data) - 5):
        seq = data[i-60:i]
        target = data[i:i+5, 0]  # Next 5 days close prices
        sequences.append(seq)
        targets.append(target)
    
    return {
        'sequences': torch.FloatTensor(sequences),
        'targets': torch.FloatTensor(targets)
    }


@pytest.fixture
def test_batch(test_sequences):
    """Create test batch data."""
    sequences = test_sequences['sequences'][:16]  # First 16 sequences
    targets = test_sequences['targets'][:16]
    
    return {
        'input': sequences,
        'target': targets,
        'batch_size': 16
    }


# Mock external dependencies
@pytest.fixture
def mock_yfinance(monkeypatch):
    """Mock yfinance for data collection tests."""
    mock_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 103, 104],
        'Low': [99, 100, 101],
        'Close': [100.5, 102.5, 103.5],
        'Volume': [1000000, 1100000, 950000],
        'Adj Close': [100.5, 102.5, 103.5]
    })
    
    def mock_download(*args, **kwargs):
        return mock_data
    
    monkeypatch.setattr('yfinance.download', mock_download)
    return mock_data


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock W&B for training tests."""
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = MagicMock()
    mock_wandb.log = MagicMock()
    mock_wandb.finish = MagicMock()
    
    monkeypatch.setattr('wandb.init', mock_wandb.init)
    monkeypatch.setattr('wandb.log', mock_wandb.log)
    monkeypatch.setattr('wandb.finish', mock_wandb.finish)
    
    return mock_wandb


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for cache tests."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    
    def mock_redis_from_url(*args, **kwargs):
        return mock_redis
    
    monkeypatch.setattr('redis.from_url', mock_redis_from_url)
    return mock_redis


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=[10, 50, 100])
def sequence_lengths(request):
    """Parametrized fixture for testing different sequence lengths."""
    return request.param


@pytest.fixture(params=[1, 5, 10])
def prediction_horizons(request):
    """Parametrized fixture for testing different prediction horizons."""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def batch_sizes(request):
    """Parametrized fixture for testing different batch sizes."""
    return request.param


@pytest.fixture(params=[1, 4, 8])
def attention_heads(request):
    """Parametrized fixture for testing different attention head counts."""
    return request.param


# Performance testing fixtures
@pytest.fixture
def performance_data():
    """Generate larger dataset for performance testing."""
    torch.manual_seed(42)
    batch_size = 64
    seq_len = 60
    features = 7
    
    return {
        'input': torch.randn(batch_size, seq_len, features),
        'target': torch.randn(batch_size, 5)
    }


@pytest.fixture
def gpu_available():
    """Check if GPU is available for GPU-specific tests."""
    return torch.cuda.is_available()


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Test data factories
class StockDataFactory:
    """Factory for generating test stock data."""
    
    @staticmethod
    def create_ohlcv_data(n_days: int = 100, base_price: float = 100.0) -> pd.DataFrame:
        """Create OHLCV stock data."""
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices)
        
        return pd.DataFrame({
            'Date': dates,
            'Open': prices * np.random.uniform(0.99, 1.01, n_days),
            'High': prices * np.random.uniform(1.005, 1.02, n_days),
            'Low': prices * np.random.uniform(0.98, 0.995, n_days),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, n_days).astype(int),
            'Adj Close': prices
        }).set_index('Date')
    
    @staticmethod
    def create_sequence_batch(batch_size: int = 32, seq_len: int = 60, 
                            n_features: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create batch of sequences for testing."""
        torch.manual_seed(42)
        inputs = torch.randn(batch_size, seq_len, n_features)
        targets = torch.randn(batch_size, 5)
        return inputs, targets


# Utility functions for tests
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_no_nan_inf(tensor: torch.Tensor):
    """Assert tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values"


def assert_gradient_flow(model: torch.nn.Module, input_tensor: torch.Tensor):
    """Assert gradients flow through model."""
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    assert input_tensor.grad is not None, "No gradients computed"
    assert not torch.isnan(input_tensor.grad).any(), "Gradient contains NaN"
    assert not torch.isinf(input_tensor.grad).any(), "Gradient contains Inf"


# Performance monitoring utilities
@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import tracemalloc
    
    class MemoryTracker:
        def __init__(self):
            self.snapshots = []
        
        def start(self):
            tracemalloc.start()
        
        def snapshot(self, label: str = ""):
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot))
        
        def stop(self):
            tracemalloc.stop()
        
        def get_peak_memory(self) -> int:
            if not self.snapshots:
                return 0
            
            peak = 0
            for _, snapshot in self.snapshots:
                current = sum(stat.size for stat in snapshot.statistics('filename'))
                peak = max(peak, current)
            
            return peak
    
    return MemoryTracker()


# Financial metrics test fixtures
@pytest.fixture
def sample_returns():
    """Generate sample returns for financial metrics testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)  # One year of daily returns


@pytest.fixture
def sample_predictions_and_actuals():
    """Generate sample predictions vs actuals for metrics testing."""
    np.random.seed(42)
    actuals = np.random.normal(0.001, 0.02, 100)
    # Add some correlation but with noise
    predictions = actuals * 0.7 + np.random.normal(0, 0.01, 100)
    
    return {
        'predictions': predictions,
        'actuals': actuals
    }


@pytest.fixture
def quantile_predictions():
    """Generate sample quantile predictions for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate quantiles that maintain ordering
    q10 = np.random.normal(-0.02, 0.005, n_samples)
    q25 = q10 + np.random.uniform(0.005, 0.01, n_samples)
    q50 = q25 + np.random.uniform(0.005, 0.01, n_samples)
    q75 = q50 + np.random.uniform(0.005, 0.01, n_samples)
    q90 = q75 + np.random.uniform(0.005, 0.01, n_samples)
    
    return {
        'quantiles': np.column_stack([q10, q25, q50, q75, q90]),
        'actuals': np.random.normal(0.001, 0.015, n_samples)
    }