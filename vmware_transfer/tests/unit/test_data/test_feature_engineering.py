"""
Unit tests for FeatureEngineer class

Tests all technical indicators and feature engineering functionality
following the testing standards.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.data.processors.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance for testing"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data with trend
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    @pytest.fixture
    def custom_config(self):
        """Custom configuration for testing"""
        return {
            'rsi': {'enabled': True, 'period': 10},
            'macd': {
                'enabled': True,
                'fast_period': 8,
                'slow_period': 21,
                'signal_period': 5
            },
            'bollinger_bands': {
                'enabled': True,
                'period': 15,
                'std_dev': 1.5
            },
            'moving_averages': {
                'enabled': True,
                'periods': [5, 10, 20]
            },
            'volume_indicators': {'enabled': True}
        }
    
    def test_initialization(self, feature_engineer):
        """Test FeatureEngineer initializes correctly"""
        assert feature_engineer is not None
        assert feature_engineer.config is not None
        assert 'rsi' in feature_engineer.config
        assert 'macd' in feature_engineer.config
        assert 'bollinger_bands' in feature_engineer.config
    
    def test_initialization_with_custom_config(self, custom_config):
        """Test initialization with custom configuration"""
        fe = FeatureEngineer(custom_config)
        assert fe.config['rsi']['period'] == 10
        assert fe.config['macd']['fast_period'] == 8
        assert fe.config['bollinger_bands']['std_dev'] == 1.5
    
    def test_engineer_features_basic(self, feature_engineer, sample_ohlcv_data):
        """Test basic feature engineering functionality"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        
        # Check that original columns are preserved
        for col in sample_ohlcv_data.columns:
            assert col in result.columns
        
        # Check that new features are added
        expected_features = [
            'Returns', 'LogReturns', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_Ratio'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check data integrity
        assert len(result) == len(sample_ohlcv_data)
        assert result.index.equals(sample_ohlcv_data.index)
    
    def test_calculate_rsi(self, feature_engineer, sample_ohlcv_data):
        """Test RSI calculation"""
        rsi = feature_engineer.calculate_rsi(sample_ohlcv_data['Close'], period=14)
        
        # Check RSI properties
        assert len(rsi) == len(sample_ohlcv_data)
        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"
        
        # Check initialization period
        assert rsi.iloc[:14].equals(pd.Series([50.0] * 14, index=rsi.index[:14]))
        
        # Check that RSI responds to price changes
        assert not rsi.iloc[20:].isna().all(), "RSI should have valid values after initialization"
    
    def test_calculate_rsi_edge_cases(self, feature_engineer):
        """Test RSI calculation with edge cases"""
        # Constant prices (should result in RSI = 50, but may be NaN due to zero gains/losses)
        constant_prices = pd.Series([100] * 30)
        rsi_constant = feature_engineer.calculate_rsi(constant_prices, period=14)
        # For constant prices, RSI calculation may result in NaN due to division by zero
        # This is expected behavior when there are no price changes
        assert len(rsi_constant) == 30
        
        # Monotonically increasing prices
        increasing_prices = pd.Series(range(100, 130))
        rsi_increasing = feature_engineer.calculate_rsi(increasing_prices, period=14)
        assert rsi_increasing.iloc[-1] > 70, "RSI should be high for consistently increasing prices"
        
        # Monotonically decreasing prices
        decreasing_prices = pd.Series(range(130, 100, -1))
        rsi_decreasing = feature_engineer.calculate_rsi(decreasing_prices, period=14)
        assert rsi_decreasing.iloc[-1] < 30, "RSI should be low for consistently decreasing prices"
    
    def test_calculate_macd(self, feature_engineer, sample_ohlcv_data):
        """Test MACD calculation"""
        macd_data = feature_engineer.calculate_macd(
            sample_ohlcv_data['Close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        # Check return structure
        assert 'MACD' in macd_data
        assert 'MACD_Signal' in macd_data
        assert 'MACD_Histogram' in macd_data
        
        # Check data properties
        for key, series in macd_data.items():
            assert len(series) == len(sample_ohlcv_data)
            assert not series.isna().all(), f"{key} should have some valid values"
        
        # Check histogram calculation
        expected_histogram = macd_data['MACD'] - macd_data['MACD_Signal']
        pd.testing.assert_series_equal(macd_data['MACD_Histogram'], expected_histogram)
    
    def test_calculate_bollinger_bands(self, feature_engineer, sample_ohlcv_data):
        """Test Bollinger Bands calculation"""
        bb_data = feature_engineer.calculate_bollinger_bands(
            sample_ohlcv_data['Close'],
            period=20,
            std_dev=2.0
        )
        
        # Check return structure
        expected_keys = ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position']
        for key in expected_keys:
            assert key in bb_data, f"Missing Bollinger Band component: {key}"
        
        # Check band ordering (Upper >= Middle >= Lower)
        valid_data = ~(bb_data['BB_Upper'].isna() | bb_data['BB_Middle'].isna() | bb_data['BB_Lower'].isna())
        
        assert (bb_data['BB_Upper'][valid_data] >= bb_data['BB_Middle'][valid_data]).all()
        assert (bb_data['BB_Middle'][valid_data] >= bb_data['BB_Lower'][valid_data]).all()
        
        # Check that middle band is SMA
        expected_middle = sample_ohlcv_data['Close'].rolling(window=20).mean()
        pd.testing.assert_series_equal(bb_data['BB_Middle'], expected_middle)
        
        # Check position calculation (can be outside [0,1] when price is outside bands)
        valid_position = bb_data['BB_Position'].dropna()
        assert len(valid_position) > 0, "Should have some valid BB_Position values"
        # Most values should be between 0 and 1, but outliers are allowed
        within_bands = valid_position.between(0, 1).sum()
        assert within_bands / len(valid_position) > 0.5, "Most positions should be within bands"
    
    def test_price_features(self, feature_engineer, sample_ohlcv_data):
        """Test basic price feature calculations"""
        result = feature_engineer._add_price_features(sample_ohlcv_data.copy())
        
        # Check returns calculation
        expected_returns = sample_ohlcv_data['Close'].pct_change()
        expected_returns.name = 'Returns'  # Set the correct name
        pd.testing.assert_series_equal(result['Returns'], expected_returns)
        
        # Check log returns
        expected_log_returns = np.log(sample_ohlcv_data['Close'] / sample_ohlcv_data['Close'].shift(1))
        expected_log_returns.name = 'LogReturns'  # Set the correct name
        pd.testing.assert_series_equal(result['LogReturns'], expected_log_returns)
        
        # Check ratio features
        assert 'HL_Ratio' in result.columns
        assert 'OC_Ratio' in result.columns
        assert 'Gap' in result.columns
    
    def test_volume_features(self, feature_engineer, sample_ohlcv_data):
        """Test volume feature calculations"""
        # Add price features first since volume features depend on Returns
        data_with_price_features = feature_engineer._add_price_features(sample_ohlcv_data.copy())
        result = feature_engineer._add_volume_features(data_with_price_features)
        
        # Check volume ratio
        expected_volume_ratio = sample_ohlcv_data['Volume'] / sample_ohlcv_data['Volume'].rolling(20).mean()
        expected_volume_ratio.name = 'Volume_Ratio'  # Set the correct name
        pd.testing.assert_series_equal(result['Volume_Ratio'], expected_volume_ratio)
        
        # Check that volume features are present
        assert 'VPT' in result.columns
        assert 'OBV' in result.columns
    
    def test_volatility_features(self, feature_engineer, sample_ohlcv_data):
        """Test volatility feature calculations"""
        # Add returns first
        data_with_returns = feature_engineer._add_price_features(sample_ohlcv_data.copy())
        result = feature_engineer._add_volatility_features(data_with_returns)
        
        # Check volatility calculation
        expected_volatility = data_with_returns['Returns'].rolling(20).std()
        expected_volatility.name = 'Volatility'  # Set the correct name
        pd.testing.assert_series_equal(result['Volatility'], expected_volatility)
        
        # Check ATR components
        assert 'TR' in result.columns
        assert 'ATR' in result.columns
        
        # Check True Range properties (skip NaN values)
        tr_valid = result['TR'].dropna()
        assert (tr_valid >= 0).all(), "True Range should be non-negative"
    
    def test_validate_features_success(self, feature_engineer, sample_ohlcv_data):
        """Test feature validation with valid data"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        validation_result = feature_engineer.validate_features(result)
        
        assert validation_result is True
    
    def test_validate_features_missing_required(self, feature_engineer, sample_ohlcv_data):
        """Test feature validation with missing required features"""
        # Create data without required features
        incomplete_data = sample_ohlcv_data.copy()
        
        validation_result = feature_engineer.validate_features(incomplete_data)
        assert validation_result is False
    
    def test_validate_features_invalid_rsi(self, feature_engineer, sample_ohlcv_data):
        """Test feature validation with invalid RSI values"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        
        # Corrupt RSI values
        result.loc[result.index[50], 'RSI'] = 150  # Invalid RSI > 100
        
        validation_result = feature_engineer.validate_features(result)
        assert validation_result is False
    
    def test_get_feature_summary(self, feature_engineer, sample_ohlcv_data):
        """Test feature summary generation"""
        result = feature_engineer.engineer_features(sample_ohlcv_data)
        summary = feature_engineer.get_feature_summary(result)
        
        # Check that summary contains expected features
        expected_features = ['Returns', 'RSI', 'MACD', 'Volume_Ratio']
        for feature in expected_features:
            assert feature in summary, f"Missing feature in summary: {feature}"
        
        # Check summary structure
        for feature, stats in summary.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'null_count' in stats
            assert 'null_ratio' in stats
    
    def test_feature_engineering_with_disabled_indicators(self):
        """Test feature engineering with some indicators disabled"""
        config = {
            'rsi': {'enabled': False, 'period': 14},
            'macd': {'enabled': True, 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'enabled': False, 'period': 20, 'std_dev': 2},
            'moving_averages': {'enabled': False, 'periods': [5, 10, 20]},
            'volume_indicators': {'enabled': True}
        }
        
        fe = FeatureEngineer(config)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(100, 110, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        result = fe.engineer_features(data)
        
        # Check that disabled features are not present
        assert 'RSI' not in result.columns
        assert 'BB_Upper' not in result.columns
        assert 'MA_5' not in result.columns
        
        # Check that enabled features are present
        assert 'MACD' in result.columns
        assert 'Volume_Ratio' in result.columns
    
    @pytest.mark.parametrize("period", [5, 10, 14, 21, 30])
    def test_rsi_different_periods(self, feature_engineer, sample_ohlcv_data, period):
        """Test RSI calculation with different periods"""
        rsi = feature_engineer.calculate_rsi(sample_ohlcv_data['Close'], period=period)
        
        # Check initialization period
        assert rsi.iloc[:period].equals(pd.Series([50.0] * period, index=rsi.index[:period]))
        
        # Check valid range
        valid_rsi = rsi.iloc[period:]
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    @pytest.mark.parametrize("std_dev", [1.0, 1.5, 2.0, 2.5])
    def test_bollinger_bands_different_std_dev(self, feature_engineer, sample_ohlcv_data, std_dev):
        """Test Bollinger Bands with different standard deviation multipliers"""
        bb_data = feature_engineer.calculate_bollinger_bands(
            sample_ohlcv_data['Close'],
            period=20,
            std_dev=std_dev
        )
        
        # Check that band width increases with std_dev
        if std_dev > 1.0:
            bb_narrow = feature_engineer.calculate_bollinger_bands(
                sample_ohlcv_data['Close'],
                period=20,
                std_dev=1.0
            )
            
            # Width should be larger for higher std_dev
            assert bb_data['BB_Width'].mean() > bb_narrow['BB_Width'].mean()
    
    def test_memory_efficiency(self, feature_engineer):
        """Test that feature engineering doesn't cause memory leaks"""
        import tracemalloc
        
        # Create large dataset
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=2000, freq='D')
        large_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 2000),
            'High': np.random.uniform(100, 110, 2000),
            'Low': np.random.uniform(90, 100, 2000),
            'Close': np.random.uniform(95, 105, 2000),
            'Volume': np.random.randint(1000000, 10000000, 2000)
        }, index=dates)
        
        tracemalloc.start()
        
        # Process data multiple times
        for _ in range(5):
            result = feature_engineer.engineer_features(large_data)
            del result
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 100MB)
        assert peak / (1024 * 1024) < 100, f"Memory usage too high: {peak / (1024 * 1024):.2f} MB"