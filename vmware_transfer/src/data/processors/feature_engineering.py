"""
Feature Engineering Module for Time-Series Transformer

This module implements technical indicators and feature engineering
for stock market data following the data pipeline standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for stock market data.
    
    Implements technical indicators including RSI, MACD, Bollinger Bands,
    and other derived features following standard financial calculations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for technical indicators."""
        return {
            'rsi': {'enabled': True, 'period': 14},
            'macd': {
                'enabled': True,
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'bollinger_bands': {
                'enabled': True,
                'period': 20,
                'std_dev': 2
            },
            'moving_averages': {
                'enabled': True,
                'periods': [5, 10, 20, 50, 200]
            },
            'volume_indicators': {'enabled': True}
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the given stock data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {len(data)} data points")
        
        # Create a copy to avoid modifying original data
        result = data.copy()
        
        # Basic price-based features
        result = self._add_price_features(result)
        
        # Technical indicators
        if self.config['rsi']['enabled']:
            result['RSI'] = self.calculate_rsi(
                result['Close'], 
                period=self.config['rsi']['period']
            )
            
        if self.config['macd']['enabled']:
            macd_data = self.calculate_macd(
                result['Close'],
                fast_period=self.config['macd']['fast_period'],
                slow_period=self.config['macd']['slow_period'],
                signal_period=self.config['macd']['signal_period']
            )
            result['MACD'] = macd_data['MACD']
            result['MACD_Signal'] = macd_data['MACD_Signal']
            result['MACD_Histogram'] = macd_data['MACD_Histogram']
            
        if self.config['bollinger_bands']['enabled']:
            bb_data = self.calculate_bollinger_bands(
                result['Close'],
                period=self.config['bollinger_bands']['period'],
                std_dev=self.config['bollinger_bands']['std_dev']
            )
            result['BB_Upper'] = bb_data['BB_Upper']
            result['BB_Middle'] = bb_data['BB_Middle']
            result['BB_Lower'] = bb_data['BB_Lower']
            result['BB_Width'] = bb_data['BB_Width']
            result['BB_Position'] = bb_data['BB_Position']
            
        # Moving averages
        if self.config['moving_averages']['enabled']:
            for period in self.config['moving_averages']['periods']:
                result[f'MA_{period}'] = result['Close'].rolling(window=period).mean()
                
        # Volume indicators
        if self.config['volume_indicators']['enabled']:
            result = self._add_volume_features(result)
            
        # Volatility features
        result = self._add_volatility_features(result)
        
        logger.info(f"Feature engineering complete. Added {len(result.columns) - len(data.columns)} features")
        
        return result
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Returns
        data['Returns'] = data['Close'].pct_change()
        data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price ranges
        data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['OC_Ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        # Gap features
        data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume ratio to moving average
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Volume-price trend
        data['VPT'] = (data['Volume'] * data['Returns']).cumsum()
        
        # On-balance volume
        data['OBV'] = (data['Volume'] * np.sign(data['Returns'])).cumsum()
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Rolling volatility
        data['Volatility'] = data['Returns'].rolling(20).std()
        
        # True range and ATR
        data['TR'] = self._calculate_true_range(data)
        data['ATR'] = data['TR'].rolling(14).mean()
        
        return data
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range for ATR calculation."""
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
        low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
        
        return np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI calculation with proper initialization
        
        Args:
            prices: Close prices series
            period: RSI period (default 14)
        
        Returns:
            RSI values [0, 100]
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle initialization period
        rsi[:period] = 50  # Neutral RSI for initial period
        
        return rsi
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD calculation with signal line and histogram
        
        Args:
            prices: Close prices series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
        
        Returns:
            Dictionary with MACD, MACD_Signal, and MACD_Histogram
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        macd_signal = macd.ewm(span=signal_period).mean()
        
        # Histogram
        macd_histogram = macd - macd_signal
        
        return {
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'MACD_Histogram': macd_histogram
        }
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands calculation
        
        Args:
            prices: Close prices series
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
        
        Returns:
            Dictionary with BB_Upper, BB_Middle, BB_Lower, BB_Width, BB_Position
        """
        # Middle band (SMA)
        bb_middle = prices.rolling(window=period).mean()
        
        # Standard deviation
        bb_std = prices.rolling(window=period).std()
        
        # Upper and lower bands
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        
        # Band width (volatility measure)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Position within bands (0 = lower band, 1 = upper band)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        
        return {
            'BB_Upper': bb_upper,
            'BB_Middle': bb_middle,
            'BB_Lower': bb_lower,
            'BB_Width': bb_width,
            'BB_Position': bb_position
        }
    
    def calculate_moving_averages(
        self, 
        prices: pd.Series, 
        periods: List[int]
    ) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages
        
        Args:
            prices: Close prices series
            periods: List of periods for moving averages
        
        Returns:
            Dictionary with moving averages for each period
        """
        mas = {}
        for period in periods:
            mas[f'MA_{period}'] = prices.rolling(window=period).mean()
        
        return mas
    
    def validate_features(self, data: pd.DataFrame) -> bool:
        """
        Validate engineered features for quality and consistency
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check for required columns
            required_features = ['Returns', 'LogReturns']
            if self.config['rsi']['enabled']:
                required_features.append('RSI')
            if self.config['macd']['enabled']:
                required_features.extend(['MACD', 'MACD_Signal'])
            if self.config['bollinger_bands']['enabled']:
                required_features.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
            
            missing_features = [f for f in required_features if f not in data.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return False
            
            # Validate RSI range
            if 'RSI' in data.columns:
                rsi_valid = data['RSI'].between(0, 100, inclusive='both').all()
                if not rsi_valid:
                    logger.error("RSI values outside valid range [0, 100]")
                    return False
            
            # Validate Bollinger Bands ordering (skip NaN values)
            if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                valid_bb_data = ~(
                    data['BB_Upper'].isna() | 
                    data['BB_Middle'].isna() | 
                    data['BB_Lower'].isna()
                )
                if valid_bb_data.any():
                    bb_valid = (
                        (data.loc[valid_bb_data, 'BB_Upper'] >= data.loc[valid_bb_data, 'BB_Middle']).all() and
                        (data.loc[valid_bb_data, 'BB_Middle'] >= data.loc[valid_bb_data, 'BB_Lower']).all()
                    )
                    if not bb_valid:
                        logger.error("Bollinger Bands ordering invalid")
                        return False
            
            # Check for excessive NaN values
            nan_ratio = data.isnull().sum() / len(data)
            max_nan_ratio = 0.1  # Allow up to 10% NaN values
            
            problematic_features = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()
            if problematic_features:
                logger.warning(f"Features with high NaN ratio: {problematic_features}")
            
            logger.info("Feature validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for engineered features
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            Dictionary with feature statistics
        """
        summary = {}
        
        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                continue  # Skip original OHLCV columns
                
            summary[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'null_count': int(data[col].isnull().sum()),
                'null_ratio': float(data[col].isnull().sum() / len(data))
            }
        
        return summary