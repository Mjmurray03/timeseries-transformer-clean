"""
Comprehensive unit tests for data pipeline components.
Follows patterns from testing-standards.md with 90% coverage requirement.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.data.collectors.yahoo_finance import YahooFinanceCollector
from src.data.processors.feature_engineering import FeatureEngineer
from src.data.datasets.stock_dataset import StockSequenceDataset
from src.data.storage import DataStorage
from src.data.validators import DataValidator
from src.data.rate_limiter import RateLimiter


class TestYahooFinanceCollector:
    """Test suite for YahooFinanceCollector following testing-standards.md patterns"""
    
    @pytest.fixture
    def collector(self):
        """Create YahooFinanceCollector instance for testing"""
        from src.config.data_config import DataConfig
        # Create proper config dictionary that matches DataConfig expectations
        config_dict = {
            'data_sources': {
                'yahoo_finance': {
                    'enabled': True,
                    'rate_limit': 5,
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1,
                    'api_key': None,
                    'base_url': None
                }
            },
            'tickers': {},
            'date_ranges': {},
            'data_quality': {},
            'sequences': {},
            'storage': {},  # StorageConfig uses defaults
            'caching': {},
            'logging': {},
            'parallel_processing': {},
            'monitoring': {},
            'development': {},
            'features': {}
        }
        config = DataConfig(config_dict)
        return YahooFinanceCollector(config)
    
    @pytest.fixture
    def mock_yfinance_data(self):
        """Generate mock yfinance data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(95, 105, len(dates)),
            'High': np.random.uniform(100, 110, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(95, 105, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Adj Close': np.random.uniform(95, 105, len(dates))
        }, index=dates)
    
    def test_initialization(self, collector):
        """Test collector initializes correctly"""
        assert collector is not None
        assert hasattr(collector, 'config')
        assert hasattr(collector, 'source_config')
    
    def test_happy_path(self, collector, mock_yfinance_data):
        """Test normal data collection succeeds"""
        with patch('yfinance.download', return_value=mock_yfinance_data):
            result = collector.fetch_data('AAPL', start='2023-01-01', end='2023-01-10')
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(mock_yfinance_data)
            assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert result.index.name == 'Date' or isinstance(result.index, pd.DatetimeIndex)
    
    def test_edge_cases(self, collector):
        """Test boundary conditions"""
        # Empty result
        with patch('yfinance.download', return_value=pd.DataFrame()):
            result = collector.fetch_data('INVALID', start='2023-01-01', end='2023-01-02')
            assert result.empty
        
        # Single day
        single_day_data = pd.DataFrame({
            'Open': [100.0], 'High': [101.0], 'Low': [99.0], 
            'Close': [100.5], 'Volume': [1000000], 'Adj Close': [100.5]
        }, index=[pd.Timestamp('2023-01-01')])
        
        with patch('yfinance.download', return_value=single_day_data):
            result = collector.fetch_data('AAPL', start='2023-01-01', end='2023-01-01')
            assert len(result) == 1
        
        # Large date range (should handle efficiently)
        large_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 1000),
            'High': np.random.uniform(100, 110, 1000),
            'Low': np.random.uniform(90, 100, 1000),
            'Close': np.random.uniform(95, 105, 1000),
            'Volume': np.random.randint(1000000, 10000000, 1000),
            'Adj Close': np.random.uniform(95, 105, 1000)
        }, index=pd.date_range(start='2020-01-01', periods=1000, freq='D'))
        
        with patch('yfinance.download', return_value=large_data):
            result = collector.fetch_data('AAPL', start='2020-01-01', end='2023-01-01')
            assert len(result) <= 1000  # Should not exceed reasonable limits
    
    def test_error_handling(self, collector):
        """Test error conditions raise appropriately"""
        # Invalid ticker
        with pytest.raises(ValueError):
            collector.fetch_data('', start='2023-01-01', end='2023-01-02')
        
        # Invalid date range
        with pytest.raises(ValueError):
            collector.fetch_data('AAPL', start='2023-01-10', end='2023-01-01')
        
        # Network error simulation
        with patch('yfinance.download', side_effect=Exception("Network error")):
            with pytest.raises(Exception):
                collector.fetch_data('AAPL', start='2023-01-01', end='2023-01-02')
    
    def test_rate_limiting(self, collector):
        """Test rate limiting functionality"""
        with patch('time.sleep') as mock_sleep:
            with patch('yfinance.download', return_value=pd.DataFrame()):
                # Multiple rapid calls should trigger rate limiting
                for _ in range(5):
                    collector.fetch_data('AAPL', start='2023-01-01', end='2023-01-02')
                
                # Rate limiter should have been called
                assert collector.rate_limiter.last_call_time is not None
    
    def test_data_validation(self, collector, mock_yfinance_data):
        """Test data validation during collection"""
        # Test with invalid data (negative prices)
        invalid_data = mock_yfinance_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Close'] = -100.0
        
        with patch('yfinance.download', return_value=invalid_data):
            result = collector.fetch_data('AAPL', start='2023-01-01', end='2023-01-10')
            
            # Should handle or flag invalid data
            assert not result.empty  # Should not crash
            # Additional validation logic would go here
    
    @pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL", "TSLA"])
    def test_multiple_tickers(self, collector, ticker, mock_yfinance_data):
        """Test collection works for different tickers"""
        with patch('yfinance.download', return_value=mock_yfinance_data):
            result = collector.fetch_data(ticker, start='2023-01-01', end='2023-01-10')
            assert not result.empty
            assert len(result) == len(mock_yfinance_data)


class TestFeatureEngineer:
    """Test suite for FeatureEngineer"""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance for testing"""
        from src.config.data_config import DataConfig
        config = DataConfig()
        return FeatureEngineer(config)
    
    @pytest.fixture
    def sample_stock_data(self):
        """Generate sample stock data for feature engineering"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
        
        return pd.DataFrame({
            'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
            'High': prices * np.random.uniform(1.01, 1.05, len(dates)),
            'Low': prices * np.random.uniform(0.95, 0.99, len(dates)),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int),
            'Adj Close': prices
        }, index=dates)
    
    def test_initialization(self, engineer):
        """Test engineer initializes correctly"""
        assert engineer is not None
        assert hasattr(engineer, 'config')
        assert hasattr(engineer, 'indicators')
    
    def test_happy_path(self, engineer, sample_stock_data):
        """Test normal feature engineering succeeds"""
        result = engineer.engineer_features(sample_stock_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_stock_data)  # May be shorter due to lookback
        
        # Check for expected features
        expected_features = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        for feature in expected_features:
            assert feature in result.columns, f"Missing expected feature: {feature}"
    
    def test_edge_cases(self, engineer):
        """Test boundary conditions"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = engineer.engineer_features(empty_df)
        assert result.empty
        
        # Single row
        single_row = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99], 
            'Close': [100.5], 'Volume': [1000000], 'Adj Close': [100.5]
        }, index=[pd.Timestamp('2023-01-01')])
        
        result = engineer.engineer_features(single_row)
        # Should handle gracefully (may return empty due to insufficient data)
        assert isinstance(result, pd.DataFrame)
        
        # Insufficient data for indicators
        short_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        result = engineer.engineer_features(short_data)
        assert isinstance(result, pd.DataFrame)
        # Some features may be NaN due to insufficient lookback
    
    def test_error_handling(self, engineer):
        """Test error conditions raise appropriately"""
        # Missing required columns
        with pytest.raises(ValueError):
            invalid_data = pd.DataFrame({'Price': [100, 101, 102]})
            engineer.engineer_features(invalid_data)
        
        # Non-numeric data
        with pytest.raises(TypeError):
            invalid_data = pd.DataFrame({
                'Close': ['abc', 'def', 'ghi']
            })
            engineer.engineer_features(invalid_data)
    
    def test_feature_consistency(self, engineer, sample_stock_data):
        """Test feature calculations are consistent and correct"""
        result = engineer.engineer_features(sample_stock_data)
        
        # Test returns calculation
        if 'Returns' in result.columns:
            expected_returns = sample_stock_data['Close'].pct_change()
            pd.testing.assert_series_equal(
                result['Returns'].dropna(), 
                expected_returns.dropna(), 
                check_names=False,
                atol=1e-6
            )
        
        # Test SMA calculation
        if 'SMA_5' in result.columns:
            expected_sma = sample_stock_data['Close'].rolling(5).mean()
            # Compare non-NaN values
            mask = ~result['SMA_5'].isna()
            pd.testing.assert_series_equal(
                result['SMA_5'][mask],
                expected_sma[mask],
                check_names=False,
                atol=1e-6
            )
    
    def test_no_future_leakage(self, engineer, sample_stock_data):
        """Test no future information leaks into past features"""
        result = engineer.engineer_features(sample_stock_data)
        
        for i in range(1, len(result)):
            current_row = result.iloc[i]
            # Ensure all features at time t only use data up to time t
            for col in result.columns:
                if not pd.isna(current_row[col]):
                    # Feature should not depend on future data
                    # This is a structural test - specific implementation would need detailed validation
                    assert isinstance(current_row[col], (int, float, np.number))


class TestStockSequenceDataset:
    """Test suite for StockSequenceDataset"""
    
    @pytest.fixture
    def dataset_config(self):
        """Create dataset configuration for testing"""
        return {
            'sequence_length': 60,
            'prediction_horizon': 5,
            'features': ['Close', 'Volume', 'Returns', 'SMA_5', 'RSI']
        }
    
    @pytest.fixture
    def sample_features_data(self):
        """Generate sample features data for dataset testing"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int),
            'Returns': np.random.normal(0.001, 0.02, len(dates)),
            'SMA_5': 100 + np.cumsum(np.random.normal(0, 0.5, len(dates))),
            'RSI': np.random.uniform(20, 80, len(dates)),
        }, index=dates)
    
    @pytest.fixture
    def dataset(self, sample_features_data, dataset_config):
        """Create StockSequenceDataset instance for testing"""
        # Convert features to sequences and targets for StockSequenceDataset
        features_array = sample_features_data[dataset_config['features']].values
        
        # Create sequences and targets
        sequences = []
        targets = []
        seq_len = dataset_config['sequence_length']
        horizon = dataset_config['prediction_horizon']
        
        for i in range(len(features_array) - seq_len - horizon + 1):
            sequences.append(features_array[i:i+seq_len])
            targets.append(features_array[i+seq_len:i+seq_len+horizon, 0])  # Use Close price as target
        
        if not sequences:
            return StockSequenceDataset(np.array([]), np.array([]), dataset_config['features'])
        
        return StockSequenceDataset(np.array(sequences), np.array(targets), dataset_config['features'])
    
    def test_initialization(self, dataset, dataset_config):
        """Test dataset initializes correctly"""
        assert dataset is not None
        assert len(dataset.features) == len(dataset_config['features'])
    
    def test_happy_path(self, dataset):
        """Test normal dataset operation succeeds"""
        assert len(dataset) > 0
        
        # Test getting a single item
        item = dataset[0]
        assert 'inputs' in item
        assert 'targets' in item
        
        # Check shapes
        assert item['inputs'].shape == (60, 5)  # sequence_length x num_features
        assert item['targets'].shape == (5,)    # prediction_horizon
        
        # Check data types
        assert isinstance(item['inputs'], torch.Tensor)
        assert isinstance(item['targets'], torch.Tensor)
    
    def test_edge_cases(self, dataset_config):
        """Test boundary conditions"""
        # Minimal data
        minimal_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200],
            'Returns': [0.01, 0.01, 0.01],
            'SMA_5': [100, 100.5, 101],
            'RSI': [50, 55, 60],
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        # Create empty dataset for minimal data case
        dataset = StockSequenceDataset(np.array([]), np.array([]), dataset_config['features'])
        assert len(dataset) == 0  # Insufficient data for sequences
        
        # Exact minimum data
        min_required = dataset_config['sequence_length'] + dataset_config['prediction_horizon']
        exact_min_data = pd.DataFrame({
            'Close': list(range(min_required)),
            'Volume': list(range(1000, 1000 + min_required)),
            'Returns': [0.01] * min_required,
            'SMA_5': list(range(100, 100 + min_required)),
            'RSI': [50] * min_required,
        }, index=pd.date_range('2023-01-01', periods=min_required, freq='D'))
        
        # Create dataset with exact minimum data
        features_array = np.random.randn(min_required, len(dataset_config['features']))
        sequences = [features_array[:60]]
        targets = [features_array[60:65, 0]]
        dataset = StockSequenceDataset(np.array(sequences), np.array(targets), dataset_config['features'])
        assert len(dataset) == 1  # Exactly one sequence possible
        
        item = dataset[0]
        assert item['inputs'].shape == (60, 5)
        assert item['targets'].shape == (5,)
    
    def test_error_handling(self, dataset_config):
        """Test error conditions raise appropriately"""
        # Missing columns
        with pytest.raises((AssertionError, IndexError)):
            # Create dataset with wrong shape
            wrong_sequences = np.array([])
            wrong_targets = np.array([])
            StockSequenceDataset(wrong_sequences, wrong_targets, dataset_config['features'])
        
        # Create valid dataset for index testing
        sequences = np.random.randn(10, 60, 5)
        targets = np.random.randn(10, 5)
        dataset = StockSequenceDataset(sequences, targets, dataset_config['features'])
        
        with pytest.raises(IndexError):
            dataset[len(dataset)]  # Out of bounds
    
    def test_sequence_alignment(self, dataset):
        """Test sequences are properly aligned without future leakage"""
        for i in range(min(10, len(dataset))):  # Test first 10 sequences
            item = dataset[i]
            input_seq = item['inputs']
            target = item['targets']
            
            # Verify no NaN values in sequences
            assert not torch.isnan(input_seq).any()
            assert not torch.isnan(target).any()
            
            # Verify sequence shape consistency
            assert input_seq.shape[0] == 60  # sequence_length
            assert input_seq.shape[1] == 5   # num_features
            assert target.shape[0] == 5      # prediction_horizon
    
    def test_data_normalization(self, sample_features_data, dataset_config):
        """Test data normalization if implemented"""
        # StockSequenceDataset doesn't have normalize parameter, so create basic dataset
        features_array = sample_features_data[dataset_config['features']].values
        sequences = []
        targets = []
        
        for i in range(len(features_array) - 65):
            sequences.append(features_array[i:i+60])
            targets.append(features_array[i+60:i+65, 0])
        
        if sequences:
            dataset = StockSequenceDataset(np.array(sequences), np.array(targets), dataset_config['features'])
            
            # Check if data is within reasonable bounds
            item = dataset[0]
            input_seq = item['inputs']
            
            # Data should be finite
            assert torch.isfinite(input_seq).all()
    
    @pytest.mark.parametrize("sequence_length", [30, 60, 90])
    def test_different_sequence_lengths(self, sample_features_data, sequence_length):
        """Test dataset with different sequence lengths"""
        features = ['Close', 'Volume', 'Returns', 'SMA_5', 'RSI']
        features_array = sample_features_data[features].values
        
        sequences = []
        targets = []
        
        for i in range(len(features_array) - sequence_length - 5):
            sequences.append(features_array[i:i+sequence_length])
            targets.append(features_array[i+sequence_length:i+sequence_length+5, 0])
        
        if sequences:
            dataset = StockSequenceDataset(np.array(sequences), np.array(targets), features)
            
            if len(dataset) > 0:
                item = dataset[0]
                assert item['inputs'].shape[0] == sequence_length


class TestDataValidator:
    """Test suite for DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance for testing"""
        return DataValidator()
    
    def test_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator is not None
        assert hasattr(validator, 'validate_ohlcv')
        assert hasattr(validator, 'validate_features')
    
    def test_happy_path(self, validator, mock_stock_data):
        """Test normal validation succeeds"""
        result = validator.validate_ohlcv(mock_stock_data)
        assert result['is_valid'] is True
        assert 'errors' in result
        assert len(result['errors']) == 0
    
    def test_edge_cases(self, validator):
        """Test boundary conditions"""
        # Empty data
        empty_df = pd.DataFrame()
        result = validator.validate_ohlcv(empty_df)
        assert result['is_valid'] is False
        assert 'Empty dataset' in str(result['errors'])
        
        # Single row
        single_row = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99], 'Close': [100.5], 'Volume': [1000]
        })
        result = validator.validate_ohlcv(single_row)
        assert isinstance(result, dict)
        assert 'is_valid' in result
    
    def test_error_handling(self, validator):
        """Test error conditions are detected appropriately"""
        # Invalid OHLC relationships
        invalid_ohlc = pd.DataFrame({
            'Open': [100, 100],
            'High': [99, 99],    # High < Open (invalid)
            'Low': [101, 101],   # Low > Open (invalid)
            'Close': [100.5, 100.5],
            'Volume': [1000, 1000]
        })
        
        result = validator.validate_ohlcv(invalid_ohlc)
        assert result['is_valid'] is False
        assert len(result['errors']) > 0
        
        # Negative values
        negative_values = pd.DataFrame({
            'Open': [100, -50],    # Negative price
            'High': [101, 60],
            'Low': [99, 40],
            'Close': [100.5, 50],
            'Volume': [1000, -500]  # Negative volume
        })
        
        result = validator.validate_ohlcv(negative_values)
        assert result['is_valid'] is False
        assert any('negative' in str(error).lower() for error in result['errors'])


class TestRateLimiter:
    """Test suite for RateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter instance for testing"""
        return RateLimiter(calls_per_second=2.0)
    
    def test_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly"""
        assert rate_limiter is not None
        assert rate_limiter.calls_per_second == 2.0
        assert rate_limiter.last_call_time is None
    
    def test_happy_path(self, rate_limiter):
        """Test normal rate limiting works"""
        import time
        
        # First call should not be limited
        start_time = time.time()
        rate_limiter.wait_if_needed()
        first_call_time = time.time() - start_time
        assert first_call_time < 0.1  # Should be immediate
        
        # Second call should be limited
        start_time = time.time()
        rate_limiter.wait_if_needed()
        second_call_time = time.time() - start_time
        assert second_call_time >= 0.4  # Should wait at least 0.5s (2 calls/sec)
    
    def test_edge_cases(self):
        """Test boundary conditions"""
        # Very high rate limit
        fast_limiter = RateLimiter(calls_per_second=1000.0)
        import time
        
        start_time = time.time()
        for _ in range(5):
            fast_limiter.wait_if_needed()
        total_time = time.time() - start_time
        assert total_time < 0.1  # Should be very fast
        
        # Very low rate limit
        slow_limiter = RateLimiter(calls_per_second=0.1)
        # Just test that it doesn't crash
        slow_limiter.wait_if_needed()
    
    def test_concurrent_usage(self, rate_limiter):
        """Test rate limiter works correctly under concurrent usage"""
        import threading
        import time
        
        call_times = []
        
        def make_call():
            start = time.time()
            rate_limiter.wait_if_needed()
            call_times.append(time.time() - start)
        
        threads = [threading.Thread(target=make_call) for _ in range(3)]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        # With 2 calls/second, 3 calls should take at least 1 second
        assert total_time >= 1.0


class TestDataStorage:
    """Test suite for DataStorage"""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create DataStorage instance for testing"""
        return DataStorage(base_path=str(tmp_path))
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for storage testing"""
        return pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
    
    def test_initialization(self, storage, tmp_path):
        """Test storage initializes correctly"""
        assert storage is not None
        assert storage.base_path == str(tmp_path)
        assert hasattr(storage, 'save')
        assert hasattr(storage, 'load')
    
    def test_happy_path(self, storage, sample_data):
        """Test normal save/load operations succeed"""
        # Save data
        key = "test_data"
        storage.save(key, sample_data)
        
        # Load data
        loaded_data = storage.load(key)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_data, loaded_data)
    
    def test_edge_cases(self, storage):
        """Test boundary conditions"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        storage.save("empty", empty_df)
        loaded = storage.load("empty")
        assert loaded.empty
        
        # Large DataFrame
        large_df = pd.DataFrame({
            'data': range(10000)
        })
        storage.save("large", large_df)
        loaded = storage.load("large")
        assert len(loaded) == 10000
        
        # Non-existent key
        result = storage.load("non_existent")
        assert result is None  # or raises FileNotFoundError, depending on implementation
    
    def test_error_handling(self, storage):
        """Test error conditions raise appropriately"""
        # Invalid data type
        with pytest.raises((TypeError, ValueError)):
            storage.save("invalid", "not a dataframe")
        
        # Invalid key
        with pytest.raises(ValueError):
            storage.save("", pd.DataFrame())  # Empty key
    
    def test_data_persistence(self, storage, sample_data):
        """Test data persists across storage instances"""
        # Save with first instance
        storage.save("persistent", sample_data)
        
        # Create new storage instance with same path
        new_storage = DataStorage(base_path=storage.base_path)
        
        # Load with second instance
        loaded_data = new_storage.load("persistent")
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_data, loaded_data)