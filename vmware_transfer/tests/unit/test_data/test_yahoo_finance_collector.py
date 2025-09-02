"""
Unit tests for YahooFinanceCollector.
Tests basic functionality, error handling, retries, and data validation.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.data.collectors.yahoo_finance import (
    YahooFinanceCollector,
    YahooFinanceError,
    RateLimiter,
    DataValidator,
    download_ticker,
    download_ticker_set
)
from src.config.data_config import DataConfig


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(rate=5, period=1)
        
        assert limiter.rate == 5
        assert limiter.period == 1
        assert limiter.tokens == 5
    
    @pytest.mark.asyncio
    async def test_token_acquisition(self):
        """Test token acquisition works."""
        limiter = RateLimiter(rate=2, period=1)
        
        # Should be able to acquire 2 tokens immediately
        await limiter.acquire()
        assert limiter.tokens < 2  # Should have decreased
        
        await limiter.acquire()
        assert limiter.tokens < 1  # Should have decreased further
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self):
        """Test that rate limiting behavior works correctly."""
        limiter = RateLimiter(rate=2, period=1)
        
        # Should be able to acquire tokens up to the limit
        initial_tokens = limiter.tokens
        await limiter.acquire()
        await limiter.acquire()
        
        # Should have fewer tokens now
        assert limiter.tokens < initial_tokens
        
        # Test that tokens can be refilled over time
        import time
        time.sleep(0.1)  # Small delay to allow refill
        await limiter._refill_tokens()
        
        # Should have more tokens after refill
        assert limiter.tokens > 0


class TestDataValidator:
    """Test data validation functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock data configuration."""
        config_dict = {
            "data_quality": {
                "min_trading_days": 252,
                "max_missing_ratio": 0.05,
                "outlier_threshold": 10.0,
                "min_volume": 100000,
                "min_price": 1.0,
                "max_price": 10000.0
            }
        }
        return DataConfig(config_dict)
    
    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock data
        base_price = 100
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
        data['Low'] = np.minimum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
        
        return data
    
    def test_validator_initialization(self, mock_config):
        """Test validator initializes correctly."""
        validator = DataValidator(mock_config)
        
        assert validator.config == mock_config
        assert validator.quality_config == mock_config.data_quality
    
    def test_valid_data_passes(self, mock_config, valid_stock_data):
        """Test that valid data passes validation."""
        validator = DataValidator(mock_config)
        
        is_valid, issues = validator.validate(valid_stock_data, "AAPL")
        
        assert is_valid
        assert len(issues) == 0
    
    def test_empty_data_fails(self, mock_config):
        """Test that empty data fails validation."""
        validator = DataValidator(mock_config)
        empty_data = pd.DataFrame()
        
        is_valid, issues = validator.validate(empty_data, "AAPL")
        
        assert not is_valid
        assert "No data returned" in issues
    
    def test_missing_columns_fails(self, mock_config, valid_stock_data):
        """Test that missing required columns fails validation."""
        validator = DataValidator(mock_config)
        
        # Remove required column
        incomplete_data = valid_stock_data.drop('Volume', axis=1)
        
        is_valid, issues = validator.validate(incomplete_data, "AAPL")
        
        assert not is_valid
        assert any("Missing columns" in issue for issue in issues)
    
    def test_insufficient_data_fails(self, mock_config):
        """Test that insufficient data fails validation."""
        validator = DataValidator(mock_config)
        
        # Create data with too few days
        short_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        })
        
        is_valid, issues = validator.validate(short_data, "AAPL")
        
        assert not is_valid
        assert any("Insufficient data" in issue for issue in issues)
    
    def test_ohlc_consistency_validation(self, mock_config, valid_stock_data):
        """Test OHLC consistency validation."""
        validator = DataValidator(mock_config)
        
        # Create inconsistent data (High < Low)
        inconsistent_data = valid_stock_data.copy()
        inconsistent_data.loc[inconsistent_data.index[0], 'High'] = 50
        inconsistent_data.loc[inconsistent_data.index[0], 'Low'] = 100
        
        is_valid, issues = validator.validate(inconsistent_data, "AAPL")
        
        assert not is_valid
        assert any("High < Low" in issue for issue in issues)
    
    def test_price_range_validation(self, mock_config, valid_stock_data):
        """Test price range validation."""
        validator = DataValidator(mock_config)
        
        # Create data with price too high
        extreme_data = valid_stock_data.copy()
        extreme_data.loc[extreme_data.index[0], 'Close'] = 20000  # Above max_price
        
        is_valid, issues = validator.validate(extreme_data, "AAPL")
        
        assert not is_valid
        assert any("price too high" in issue for issue in issues)
    
    def test_volume_validation(self, mock_config, valid_stock_data):
        """Test volume validation."""
        validator = DataValidator(mock_config)
        
        # Create data with negative volume
        bad_volume_data = valid_stock_data.copy()
        bad_volume_data.loc[bad_volume_data.index[0], 'Volume'] = -1000
        
        is_valid, issues = validator.validate(bad_volume_data, "AAPL")
        
        assert not is_valid
        assert any("Negative volume" in issue for issue in issues)


class TestYahooFinanceCollector:
    """Test YahooFinanceCollector functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config_dict = {
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "rate_limit": 5,
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 1
                }
            },
            "data_quality": {
                "min_trading_days": 10,  # Reduced for testing
                "max_missing_ratio": 0.05,
                "outlier_threshold": 10.0,
                "min_volume": 100000,
                "min_price": 1.0,
                "max_price": 10000.0
            },
            "tickers": {
                "test_set": ["AAPL", "MSFT"]
            },
            "date_ranges": {
                "test_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                }
            }
        }
        return DataConfig(config_dict)
    
    @pytest.fixture
    def sample_yfinance_data(self):
        """Create sample data that yfinance would return."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        base_price = 150
        returns = np.random.normal(0, 0.01, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'Close': prices,
            'Adj Close': prices * 0.98,  # Slightly adjusted
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
        data['Low'] = np.minimum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
        
        return data
    
    def test_collector_initialization(self, mock_config):
        """Test collector initializes correctly."""
        collector = YahooFinanceCollector(mock_config)
        
        assert collector.config == mock_config
        assert collector.source_config.enabled
        assert collector.source_config.rate_limit == 5
        assert isinstance(collector.rate_limiter, RateLimiter)
        assert isinstance(collector.validator, DataValidator)
    
    def test_collector_initialization_disabled_source(self, mock_config):
        """Test collector raises error when source is disabled."""
        # Disable yahoo finance
        mock_config.data_sources['yahoo_finance'].enabled = False
        
        with pytest.raises(YahooFinanceError, match="disabled"):
            YahooFinanceCollector(mock_config)
    
    def test_collector_initialization_missing_config(self):
        """Test collector raises error when config is missing."""
        config_dict = {"data_sources": {}}  # No yahoo_finance config
        config = DataConfig(config_dict)
        
        with pytest.raises(YahooFinanceError, match="not found"):
            YahooFinanceCollector(config)
    
    @pytest.mark.asyncio
    @patch('yfinance.download')
    async def test_successful_data_collection(self, mock_download, mock_config, sample_yfinance_data):
        """Test successful data collection for a single ticker."""
        mock_download.return_value = sample_yfinance_data
        
        collector = YahooFinanceCollector(mock_config)
        
        result = await collector.collect_ticker(
            "AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Ticker' in result.columns
        assert result['Ticker'].iloc[0] == 'AAPL'
        
        # Check statistics
        stats = collector.get_statistics()
        assert stats['successful_downloads'] == 1
        assert stats['failed_downloads'] == 0
    
    @pytest.mark.asyncio
    @patch('yfinance.download')
    async def test_data_collection_with_empty_response(self, mock_download, mock_config):
        """Test handling of empty response from yfinance."""
        mock_download.return_value = pd.DataFrame()  # Empty DataFrame
        
        collector = YahooFinanceCollector(mock_config)
        
        result = await collector.collect_ticker("INVALID")
        
        assert result is None
        
        # Check statistics
        stats = collector.get_statistics()
        assert stats['failed_downloads'] == 1
    
    @pytest.mark.asyncio
    @patch('yfinance.download')
    async def test_data_collection_with_validation_failure(self, mock_download, mock_config):
        """Test handling of data that fails validation."""
        # Create invalid data (insufficient days)
        invalid_data = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000000]
        })
        mock_download.return_value = invalid_data
        
        collector = YahooFinanceCollector(mock_config)
        
        result = await collector.collect_ticker("AAPL")
        
        assert result is None
        
        # Check statistics
        stats = collector.get_statistics()
        assert stats['validation_failures'] == 1
    
    @pytest.mark.asyncio
    @patch('yfinance.download')
    async def test_multiple_ticker_collection(self, mock_download, mock_config, sample_yfinance_data):
        """Test collecting data for multiple tickers."""
        mock_download.return_value = sample_yfinance_data
        
        collector = YahooFinanceCollector(mock_config)
        
        tickers = ["AAPL", "MSFT", "GOOGL"]
        results = await collector.collect_multiple(
            tickers,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            max_concurrent=2
        )
        
        assert len(results) == 3
        for ticker in tickers:
            assert ticker in results
            assert isinstance(results[ticker], pd.DataFrame)
            assert results[ticker]['Ticker'].iloc[0] == ticker
    
    @pytest.mark.asyncio
    @patch('yfinance.download')
    async def test_retry_mechanism(self, mock_download, mock_config):
        """Test retry mechanism on failures."""
        # First call fails, second succeeds
        mock_download.side_effect = [
            Exception("Network error"),
            pd.DataFrame({
                'Open': [100] * 20,
                'High': [102] * 20,
                'Low': [99] * 20,
                'Close': [101] * 20,
                'Volume': [1000000] * 20
            }, index=pd.date_range('2023-01-01', periods=20))
        ]
        
        collector = YahooFinanceCollector(mock_config)
        
        result = await collector.collect_ticker("AAPL")
        
        assert result is not None
        
        # Should have made multiple requests due to retry
        stats = collector.get_statistics()
        assert stats['total_retries'] > 0
    
    def test_data_cleaning(self, mock_config, sample_yfinance_data):
        """Test data cleaning functionality."""
        collector = YahooFinanceCollector(mock_config)
        
        # Add some issues to test cleaning
        dirty_data = sample_yfinance_data.copy()
        
        # Add duplicate index
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:1]])
        
        # Add missing values
        dirty_data.loc[dirty_data.index[5], 'Close'] = np.nan
        
        cleaned = collector._clean_data(dirty_data, "AAPL")
        
        # Check that duplicates were removed
        assert not cleaned.index.duplicated().any()
        
        # Check that missing values were handled
        assert not cleaned.isnull().any().any()
        
        # Check that ticker was added
        assert 'Ticker' in cleaned.columns
        assert cleaned['Ticker'].iloc[0] == 'AAPL'
    
    def test_statistics_tracking(self, mock_config):
        """Test statistics tracking and reset."""
        collector = YahooFinanceCollector(mock_config)
        
        # Initial stats should be zero
        stats = collector.get_statistics()
        assert all(value == 0 for value in stats.values())
        
        # Modify stats
        collector.stats['successful_downloads'] = 5
        collector.stats['failed_downloads'] = 2
        
        # Get stats
        stats = collector.get_statistics()
        assert stats['successful_downloads'] == 5
        assert stats['failed_downloads'] == 2
        
        # Reset stats
        collector.reset_statistics()
        stats = collector.get_statistics()
        assert all(value == 0 for value in stats.values())


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config_dict = {
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "rate_limit": 5,
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 1
                }
            },
            "data_quality": {
                "min_trading_days": 10,
                "max_missing_ratio": 0.05,
                "outlier_threshold": 10.0,
                "min_volume": 100000,
                "min_price": 1.0,
                "max_price": 10000.0
            },
            "tickers": {
                "test_set": ["AAPL", "MSFT"]
            },
            "date_ranges": {
                "test_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                }
            }
        }
        return DataConfig(config_dict)
    
    @pytest.mark.asyncio
    @patch('src.data.collectors.yahoo_finance.YahooFinanceCollector')
    async def test_download_ticker_function(self, mock_collector_class, mock_config):
        """Test download_ticker convenience function."""
        # Mock the collector instance
        mock_collector = Mock()
        mock_collector.collect_ticker = AsyncMock(return_value=pd.DataFrame({'test': [1, 2, 3]}))
        mock_collector_class.return_value = mock_collector
        
        result = await download_ticker("AAPL", mock_config, "test_range")
        
        assert result is not None
        mock_collector.collect_ticker.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.data.collectors.yahoo_finance.YahooFinanceCollector')
    async def test_download_ticker_set_function(self, mock_collector_class, mock_config):
        """Test download_ticker_set convenience function."""
        # Mock the collector instance
        mock_collector = Mock()
        mock_collector.collect_multiple = AsyncMock(return_value={
            "AAPL": pd.DataFrame({'test': [1, 2, 3]}),
            "MSFT": pd.DataFrame({'test': [4, 5, 6]})
        })
        mock_collector_class.return_value = mock_collector
        
        result = await download_ticker_set("test_set", mock_config, "test_range")
        
        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        mock_collector.collect_multiple.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_ticker_invalid_date_range(self, mock_config):
        """Test download_ticker with invalid date range."""
        result = await download_ticker("AAPL", mock_config, "invalid_range")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_download_ticker_set_invalid_ticker_set(self, mock_config):
        """Test download_ticker_set with invalid ticker set."""
        result = await download_ticker_set("invalid_set", mock_config, "test_range")
        
        assert len(result) == 0


# Integration test fixtures and helpers
@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    config_dict = {
        "data_sources": {
            "yahoo_finance": {
                "enabled": True,
                "rate_limit": 1,  # Slow for integration tests
                "timeout": 30,
                "retry_attempts": 2,
                "retry_delay": 1
            }
        },
        "data_quality": {
            "min_trading_days": 5,  # Reduced for testing
            "max_missing_ratio": 0.1,
            "outlier_threshold": 10.0,
            "min_volume": 10000,  # Reduced for testing
            "min_price": 1.0,
            "max_price": 10000.0
        },
        "tickers": {
            "integration_test": ["AAPL"]  # Single ticker for integration
        },
        "date_ranges": {
            "recent": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"  # Small range for testing
            }
        }
    }
    return DataConfig(config_dict)


@pytest.mark.integration
class TestYahooFinanceIntegration:
    """Integration tests with real Yahoo Finance API."""
    
    @pytest.mark.asyncio
    async def test_real_data_download(self, integration_config):
        """Test downloading real data from Yahoo Finance."""
        collector = YahooFinanceCollector(integration_config)
        
        result = await collector.collect_ticker(
            "AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        if result is not None:  # May fail due to network issues
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'Ticker' in result.columns
            assert result['Ticker'].iloc[0] == 'AAPL'
            
            # Check that all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                assert col in result.columns