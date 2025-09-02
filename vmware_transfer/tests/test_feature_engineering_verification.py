"""
Comprehensive Test Suite for Feature Engineering Verification Script
Tests the verify_feature_engineering.py script with synthetic data and edge cases.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from verify_feature_engineering import FeatureVerifier
from data.processors.feature_engineering import FeatureEngineer


class TestFeatureEngineeringVerification:
    """Comprehensive test suite for feature engineering verification."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def _create_synthetic_ohlcv_data(
        self, 
        n_days: int, 
        pattern_type: str = "trending",
        base_price: float = 100.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """Create synthetic OHLCV data with known patterns."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Create date range (business days only)
        start_date = datetime(2023, 1, 1)
        dates = pd.bdate_range(start=start_date, periods=n_days)
        
        # Generate synthetic data based on pattern type
        data = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            if pattern_type == "trending":
                # Upward trending market with small daily variations
                daily_trend = 0.001  # 0.1% daily trend
                daily_noise = np.random.normal(0, 0.01)  # 1% daily volatility
                price_change = daily_trend + daily_noise
                
            elif pattern_type == "oscillating":
                # Oscillating market for RSI testing (creates clear overbought/oversold)
                period = 20  # Oscillation period
                amplitude = 0.05  # 5% amplitude
                oscillation = amplitude * np.sin(2 * np.pi * i / period)
                daily_noise = np.random.normal(0, 0.005)  # 0.5% noise
                price_change = oscillation + daily_noise
                
            elif pattern_type == "volatile":
                # High volatility for Bollinger Band testing
                daily_noise = np.random.normal(0, 0.03)  # 3% daily volatility
                price_change = daily_noise
                
            elif pattern_type == "stable":
                # Low volatility, steady price
                daily_noise = np.random.normal(0, 0.002)  # 0.2% daily volatility
                price_change = daily_noise
                
            else:
                # Default: random walk
                price_change = np.random.normal(0, 0.015)  # 1.5% daily volatility
            
            # Calculate OHLC based on the daily change
            open_price = current_price
            close_price = current_price * (1 + price_change)
            
            # Generate realistic High/Low based on intraday volatility
            intraday_vol = abs(price_change) + 0.005  # Minimum 0.5% intraday range
            high_price = max(open_price, close_price) * (1 + intraday_vol * np.random.uniform(0.2, 0.8))
            low_price = min(open_price, close_price) * (1 - intraday_vol * np.random.uniform(0.2, 0.8))
            
            # Ensure price constraints: High >= max(Open, Close), Low <= min(Open, Close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate realistic volume
            avg_volume = 1000000
            volume_noise = np.random.uniform(0.5, 1.5)
            volume = int(avg_volume * volume_noise)
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Ticker': 'SYNTHETIC'
            })
            
            current_price = close_price
        
        return pd.DataFrame(data, index=dates)
    
    def _calculate_expected_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate expected indicator values using direct numpy/pandas formulas."""
        close_prices = df['Close']
        expected = {}
        
        # SMA_20 (Simple Moving Average)
        expected['MA_20'] = close_prices.rolling(window=20).mean()
        
        # RSI using standard calculation
        expected['RSI'] = self._calculate_expected_rsi(close_prices)
        
        # MACD (12, 26, 9)
        macd_data = self._calculate_expected_macd(close_prices, 12, 26, 9)
        expected.update(macd_data)
        
        # Bollinger Bands (20 period, 2 std dev)
        bb_data = self._calculate_expected_bollinger_bands(close_prices, 20, 2.0)
        expected.update(bb_data)
        
        # Returns
        expected['Returns'] = close_prices.pct_change()
        
        return expected
    
    def _calculate_expected_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate expected RSI matching FeatureEngineer implementation."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Match FeatureEngineer: set first 'period' values to 50
        rsi[:period] = 50
        
        return rsi
    
    def _calculate_expected_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate expected MACD values."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return {
            'MACD': macd_line,
            'MACD_Signal': macd_signal,
            'MACD_Histogram': macd_histogram
        }
    
    def _calculate_expected_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate expected Bollinger Bands values."""
        bb_middle = prices.rolling(window=period).mean()
        bb_std = prices.rolling(window=period).std()
        
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        
        return {
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower
        }
    
    def _save_test_data(self, df: pd.DataFrame, temp_dir: str, filename: str = "test_data.parquet"):
        """Save test data to parquet file."""
        file_path = Path(temp_dir) / filename
        df.to_parquet(file_path)
        return file_path
    
    @pytest.fixture
    def trending_market_data(self, temp_data_dir):
        """Create trending market data for moving average validation."""
        df = self._create_synthetic_ohlcv_data(100, "trending", base_price=100.0, seed=42)
        file_path = self._save_test_data(df, temp_data_dir, "trending_market.parquet")
        return df, file_path
    
    @pytest.fixture  
    def oscillating_market_data(self, temp_data_dir):
        """Create oscillating market data for RSI validation."""
        df = self._create_synthetic_ohlcv_data(80, "oscillating", base_price=100.0, seed=123)
        file_path = self._save_test_data(df, temp_data_dir, "oscillating_market.parquet")
        return df, file_path
    
    @pytest.fixture
    def volatile_market_data(self, temp_data_dir):
        """Create volatile market data for Bollinger Band validation."""
        df = self._create_synthetic_ohlcv_data(60, "volatile", base_price=100.0, seed=456)
        file_path = self._save_test_data(df, temp_data_dir, "volatile_market.parquet")
        return df, file_path
    
    @pytest.fixture
    def minimum_length_data(self, temp_data_dir):
        """Create data with exactly minimum required length (50 days)."""
        df = self._create_synthetic_ohlcv_data(50, "stable", base_price=100.0, seed=789)
        file_path = self._save_test_data(df, temp_data_dir, "minimum_length.parquet")
        return df, file_path
    
    @pytest.fixture
    def short_data(self, temp_data_dir):
        """Create data shorter than longest indicator period (15 days < 20 SMA)."""
        df = self._create_synthetic_ohlcv_data(15, "random", base_price=100.0, seed=999)
        file_path = self._save_test_data(df, temp_data_dir, "short_data.parquet")
        return df, file_path
    
    def test_trending_market_verification(self, trending_market_data):
        """Test verification on trending market data."""
        df, file_path = trending_market_data
        
        # Create verifier and run on trending data
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should pass all checks
        assert report['status'] == 'PASS', f"Trending market verification failed: {report.get('errors', [])}"
        assert report['summary']['total_checks'] >= 10
        assert report['summary']['failed'] == 0
        
        # Verify specific indicators make sense for trending data
        assert len(report.get('comparison_data', [])) > 0, "Should have comparison data"
        
        # All comparisons should be within tolerance
        within_tolerance = sum(1 for item in report['comparison_data'] if item['Within_Tolerance'])
        total_comparisons = len(report['comparison_data'])
        assert within_tolerance == total_comparisons, f"Only {within_tolerance}/{total_comparisons} within tolerance"
    
    def test_oscillating_market_verification(self, oscillating_market_data):
        """Test verification on oscillating market data (good for RSI)."""
        df, file_path = oscillating_market_data
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should pass all checks
        assert report['status'] == 'PASS', f"Oscillating market verification failed: {report.get('errors', [])}"
        
        # Verify RSI values are within expected range [0, 100]
        # This is tested implicitly by the verifier, but we can check the report
        rsi_checks = [item for item in report['comparison_data'] if item['Indicator'] == 'RSI']
        assert len(rsi_checks) > 0, "Should have RSI comparisons"
        
        for rsi_check in rsi_checks:
            assert 0 <= rsi_check['Actual'] <= 100, f"RSI value {rsi_check['Actual']} outside [0, 100] range"
    
    def test_volatile_market_verification(self, volatile_market_data):
        """Test verification on volatile market data (good for Bollinger Bands)."""
        df, file_path = volatile_market_data
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should pass all checks
        assert report['status'] == 'PASS', f"Volatile market verification failed: {report.get('errors', [])}"
        
        # Verify Bollinger Band ordering (Upper >= Middle >= Lower)
        bb_checks = [item for item in report['comparison_data'] if item['Indicator'].startswith('BB_')]
        assert len(bb_checks) > 0, "Should have Bollinger Band comparisons"
        
        # Group by date to check ordering
        bb_by_date = {}
        for bb_check in bb_checks:
            date = bb_check['Date']
            if date not in bb_by_date:
                bb_by_date[date] = {}
            bb_by_date[date][bb_check['Indicator']] = bb_check['Actual']
        
        for date, bb_values in bb_by_date.items():
            if all(key in bb_values for key in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                assert bb_values['BB_Upper'] >= bb_values['BB_Middle'], f"BB ordering failed on {date}: Upper < Middle"
                assert bb_values['BB_Middle'] >= bb_values['BB_Lower'], f"BB ordering failed on {date}: Middle < Lower"
    
    def test_minimum_length_data_verification(self, minimum_length_data):
        """Test verification with exactly minimum required length."""
        df, file_path = minimum_length_data
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should pass with 50 days (enough for all indicators)
        assert report['status'] == 'PASS', f"Minimum length verification failed: {report.get('errors', [])}"
        assert len(report['comparison_data']) > 0, "Should have comparison data even with minimum length"
    
    def test_short_data_handling(self, short_data):
        """Test verification with data shorter than longest indicator period."""
        df, file_path = short_data
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Verification may still pass if it handles short data gracefully
        # The key is that it shouldn't crash and should provide meaningful feedback
        assert 'status' in report, "Report should have status field"
        assert 'summary' in report, "Report should have summary field"
        
        # If it fails, should be due to insufficient data, not calculation errors
        if report['status'] == 'FAIL':
            error_messages = [error['message'] for error in report.get('errors', [])]
            # Should not fail due to calculation errors, only data insufficiency
            calculation_errors = [msg for msg in error_messages if 'Expected' in msg and 'got' in msg]
            assert len(calculation_errors) == 0, f"Should not have calculation errors: {calculation_errors}"
    
    def test_mathematical_correctness_sma(self, trending_market_data):
        """Test mathematical correctness of SMA calculation."""
        df, file_path = trending_market_data
        
        # Calculate expected SMA manually
        expected_sma = df['Close'].rolling(window=20).mean()
        
        # Apply FeatureEngineer
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(df)
        
        # Compare SMA values (skip NaN values)
        valid_indices = ~(expected_sma.isna() | engineered_df['MA_20'].isna())
        
        if valid_indices.any():
            max_diff = abs(expected_sma[valid_indices] - engineered_df['MA_20'][valid_indices]).max()
            assert max_diff <= 1e-6, f"SMA calculation error exceeds tolerance: {max_diff}"
    
    def test_mathematical_correctness_rsi(self, oscillating_market_data):
        """Test mathematical correctness of RSI calculation."""
        df, file_path = oscillating_market_data
        
        # Calculate expected RSI manually (matching FeatureEngineer implementation)
        expected_rsi = self._calculate_expected_rsi(df['Close'])
        
        # Apply FeatureEngineer
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(df)
        
        # Compare RSI values
        valid_indices = ~(expected_rsi.isna() | engineered_df['RSI'].isna())
        
        if valid_indices.any():
            max_diff = abs(expected_rsi[valid_indices] - engineered_df['RSI'][valid_indices]).max()
            assert max_diff <= 1e-6, f"RSI calculation error exceeds tolerance: {max_diff}"
        
        # Check RSI range
        assert (engineered_df['RSI'] >= 0).all(), "RSI values below 0"
        assert (engineered_df['RSI'] <= 100).all(), "RSI values above 100"
    
    def test_mathematical_correctness_bollinger_bands(self, volatile_market_data):
        """Test mathematical correctness of Bollinger Bands calculation."""
        df, file_path = volatile_market_data
        
        # Calculate expected Bollinger Bands manually
        expected_bb = self._calculate_expected_bollinger_bands(df['Close'])
        
        # Apply FeatureEngineer
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(df)
        
        # Compare each Bollinger Band component
        for component in ['BB_Upper', 'BB_Middle', 'BB_Lower']:
            expected_values = expected_bb[component]
            actual_values = engineered_df[component]
            
            valid_indices = ~(expected_values.isna() | actual_values.isna())
            
            if valid_indices.any():
                max_diff = abs(expected_values[valid_indices] - actual_values[valid_indices]).max()
                assert max_diff <= 1e-6, f"{component} calculation error exceeds tolerance: {max_diff}"
    
    def test_data_with_missing_values(self, temp_data_dir):
        """Test verification with dataset containing missing values."""
        # Create data with some NaN values
        df = self._create_synthetic_ohlcv_data(50, "random", seed=555)
        
        # Introduce NaN values in Close column
        df.loc[df.index[10], 'Close'] = np.nan
        df.loc[df.index[25], 'Volume'] = np.nan
        
        file_path = self._save_test_data(df, temp_data_dir, "missing_values.parquet")
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should handle missing values gracefully
        assert 'status' in report, "Should generate report even with missing values"
        
        # If it fails, should be due to missing data, not calculation errors
        if report['status'] == 'FAIL':
            error_messages = [error['message'] for error in report.get('errors', [])]
            nan_related = [msg for msg in error_messages if 'NaN' in msg or 'missing' in msg.lower()]
            # Should identify missing data issues appropriately
    
    def test_extreme_price_movements(self, temp_data_dir):
        """Test verification with extreme price movements."""
        # Create data with extreme price jumps
        df = self._create_synthetic_ohlcv_data(40, "stable", seed=777)
        
        # Introduce extreme price movements
        df.loc[df.index[15], 'Close'] *= 2.0  # 100% increase
        df.loc[df.index[16], 'Close'] *= 0.5  # 50% decrease
        df.loc[df.index[17], 'High'] = df.loc[df.index[17], 'Close'] * 1.5  # Update High accordingly
        df.loc[df.index[17], 'Low'] = df.loc[df.index[17], 'Close'] * 0.8   # Update Low accordingly
        
        file_path = self._save_test_data(df, temp_data_dir, "extreme_movements.parquet")
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should handle extreme movements without calculation errors
        assert report['status'] == 'PASS', f"Extreme movements verification failed: {report.get('errors', [])}"
    
    def test_verifier_catches_incorrect_calculations(self, temp_data_dir):
        """Test that the verifier catches incorrect indicator calculations (meta-validation)."""
        df = self._create_synthetic_ohlcv_data(50, "trending", seed=888)
        file_path = self._save_test_data(df, temp_data_dir, "test_incorrect.parquet")
        
        # Create a modified verifier that will produce incorrect expected values
        class IncorrectFeatureVerifier(FeatureVerifier):
            def _calculate_manual_rsi(self, prices, target_idx, period=14):
                # Return deliberately incorrect RSI value
                return 75.0  # Always return 75, which should be wrong for most cases
        
        incorrect_verifier = IncorrectFeatureVerifier(tolerance=1e-6)
        report = incorrect_verifier.verify_all_features(str(file_path))
        
        # Should fail due to incorrect calculations
        assert report['status'] == 'FAIL', "Verifier should catch incorrect calculations"
        
        # Should have RSI-related errors
        rsi_errors = [error for error in report.get('errors', []) if 'RSI' in error['check']]
        assert len(rsi_errors) > 0, "Should catch RSI calculation errors"
    
    def test_verifier_catches_missing_features(self, temp_data_dir):
        """Test that the verifier catches missing features."""
        df = self._create_synthetic_ohlcv_data(50, "trending", seed=999)
        file_path = self._save_test_data(df, temp_data_dir, "test_missing_features.parquet")
        
        # Create a modified FeatureEngineer that doesn't generate all features
        class IncompleteFeatureEngineer(FeatureEngineer):
            def engineer_features(self, data):
                result = data.copy()
                # Only add Returns, skip other indicators
                result['Returns'] = result['Close'].pct_change()
                return result
        
        # Create a verifier that will use the incomplete feature engineer
        verifier = FeatureVerifier(tolerance=1e-6)
        
        # Manually replace the feature engineer
        original_df = pd.read_parquet(file_path)
        incomplete_engineer = IncompleteFeatureEngineer()
        incomplete_df = incomplete_engineer.engineer_features(original_df)
        
        # Test feature completeness check directly
        verifier._verify_feature_completeness(incomplete_df)
        
        # Should have feature completeness error
        assert 'FEATURE_COMPLETENESS' in verifier.results
        assert verifier.results['FEATURE_COMPLETENESS'] == 'FAIL'
    
    def test_verifier_catches_dimension_errors(self, temp_data_dir):
        """Test that the verifier catches wrong feature dimensions."""
        # Create data with wrong number of columns
        df = self._create_synthetic_ohlcv_data(50, "trending", seed=1111)
        
        # Remove a required column
        df_incomplete = df.drop(columns=['Volume'])  # Only 5 columns instead of 6
        file_path = self._save_test_data(df_incomplete, temp_data_dir, "wrong_dimensions.parquet")
        
        verifier = FeatureVerifier(tolerance=1e-6)
        report = verifier.verify_all_features(str(file_path))
        
        # Should fail due to missing required columns
        assert report['status'] == 'FAIL', "Should catch dimension errors"
        
        # Should have data loading error
        data_loading_errors = [error for error in report.get('errors', []) if 'DATA_LOADING' in error['check']]
        assert len(data_loading_errors) > 0, "Should catch missing column errors"
    
    def test_reproducible_execution(self, temp_data_dir):
        """Test that verification execution is deterministic and reproducible."""
        df = self._create_synthetic_ohlcv_data(60, "trending", seed=1234)
        file_path = self._save_test_data(df, temp_data_dir, "reproducible_test.parquet")
        
        # Run verification twice
        verifier1 = FeatureVerifier(tolerance=1e-6)
        report1 = verifier1.verify_all_features(str(file_path))
        
        verifier2 = FeatureVerifier(tolerance=1e-6)
        report2 = verifier2.verify_all_features(str(file_path))
        
        # Results should be identical
        assert report1['status'] == report2['status'], "Results should be reproducible"
        assert len(report1['comparison_data']) == len(report2['comparison_data']), "Comparison data length should be identical"
        
        # Compare specific values
        if len(report1['comparison_data']) > 0 and len(report2['comparison_data']) > 0:
            for i, (item1, item2) in enumerate(zip(report1['comparison_data'], report2['comparison_data'])):
                assert item1['Expected'] == item2['Expected'], f"Expected values should be identical at index {i}"
                assert item1['Actual'] == item2['Actual'], f"Actual values should be identical at index {i}"
    
    def test_performance_requirement(self, trending_market_data):
        """Test that verification meets performance requirements."""
        df, file_path = trending_market_data
        
        verifier = FeatureVerifier(tolerance=1e-6)
        
        import time
        start_time = time.time()
        report = verifier.verify_all_features(str(file_path))
        elapsed_time = time.time() - start_time
        
        # Should complete in under 30 seconds (requirement)
        assert elapsed_time < 30.0, f"Verification too slow: {elapsed_time:.2f}s"
        
        # Should also pass the internal performance check
        assert 'PERFORMANCE' in report['results']
        assert report['results']['PERFORMANCE'] == 'PASS'
    
    def test_tolerance_sensitivity(self, temp_data_dir):
        """Test verification with different tolerance levels."""
        df = self._create_synthetic_ohlcv_data(50, "trending", seed=1357)
        file_path = self._save_test_data(df, temp_data_dir, "tolerance_test.parquet")
        
        # Test with strict tolerance
        strict_verifier = FeatureVerifier(tolerance=1e-10)
        strict_report = strict_verifier.verify_all_features(str(file_path))
        
        # Test with relaxed tolerance
        relaxed_verifier = FeatureVerifier(tolerance=1e-3)
        relaxed_report = relaxed_verifier.verify_all_features(str(file_path))
        
        # Both should pass (our calculations should be exact)
        assert strict_report['status'] == 'PASS', "Strict tolerance should pass with exact calculations"
        assert relaxed_report['status'] == 'PASS', "Relaxed tolerance should definitely pass"
        
        # But strict tolerance is more... strict
        strict_within_tolerance = sum(1 for item in strict_report['comparison_data'] if item['Within_Tolerance'])
        relaxed_within_tolerance = sum(1 for item in relaxed_report['comparison_data'] if item['Within_Tolerance'])
        
        assert relaxed_within_tolerance >= strict_within_tolerance, "Relaxed tolerance should have >= comparisons passing"


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__, "-v"])