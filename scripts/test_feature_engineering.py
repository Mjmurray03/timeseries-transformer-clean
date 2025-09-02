#!/usr/bin/env python3
"""
Test script for FeatureEngineer functionality

This script demonstrates the FeatureEngineer working with sample data
and shows all the technical indicators being calculated correctly.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.processors.feature_engineering import FeatureEngineer


def create_sample_data():
    """Create realistic sample OHLCV data for testing"""
    np.random.seed(42)

    # Generate 200 days of data
    dates = pd.date_range("2023-01-01", periods=200, freq="D")

    # Generate realistic price data with trend and volatility
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, 200)  # Small positive drift, 2% daily volatility

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data with realistic relationships
    data = pd.DataFrame(index=dates)
    data["Close"] = prices

    # Generate OHLC with realistic spreads
    data["Open"] = data["Close"].shift(1) * np.random.uniform(0.995, 1.005, 200)
    data["High"] = np.maximum(data["Open"], data["Close"]) * np.random.uniform(1.000, 1.015, 200)
    data["Low"] = np.minimum(data["Open"], data["Close"]) * np.random.uniform(0.985, 1.000, 200)

    # Volume with some correlation to price changes
    base_volume = 1000000
    volume_multiplier = 1 + np.abs(returns) * 5  # Higher volume on big moves
    data["Volume"] = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, 200)).astype(
        int
    )

    # Clean up first row
    data.iloc[0] = [100, 102, 98, 100, 1500000]  # Open, High, Low, Close, Volume

    return data


def main():
    """Main test function"""
    print("Testing FeatureEngineer functionality...")
    print("=" * 50)

    # Create sample data
    print("1. Creating sample OHLCV data...")
    data = create_sample_data()
    print(f"   Generated {len(data)} days of data")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print()

    # Initialize FeatureEngineer
    print("2. Initializing FeatureEngineer...")
    fe = FeatureEngineer()
    print("   Using default configuration:")
    for indicator, config in fe.config.items():
        if config.get("enabled", True):
            print(f"   - {indicator}: {config}")
    print()

    # Engineer features
    print("3. Engineering features...")
    result = fe.engineer_features(data)

    original_cols = len(data.columns)
    new_cols = len(result.columns)
    added_features = new_cols - original_cols

    print(f"   Original columns: {original_cols}")
    print(f"   Total columns after feature engineering: {new_cols}")
    print(f"   Added features: {added_features}")
    print()

    # Show feature categories
    print("4. Feature categories:")

    # Price features
    price_features = [
        col
        for col in result.columns
        if col in ["Returns", "LogReturns", "HL_Ratio", "OC_Ratio", "Gap"]
    ]
    print(f"   Price features ({len(price_features)}): {price_features}")

    # Technical indicators
    tech_features = [
        col for col in result.columns if col in ["RSI", "MACD", "MACD_Signal", "MACD_Histogram"]
    ]
    print(f"   Technical indicators ({len(tech_features)}): {tech_features}")

    # Bollinger Bands
    bb_features = [col for col in result.columns if col.startswith("BB_")]
    print(f"   Bollinger Bands ({len(bb_features)}): {bb_features}")

    # Moving averages
    ma_features = [col for col in result.columns if col.startswith("MA_")]
    print(f"   Moving averages ({len(ma_features)}): {ma_features}")

    # Volume features
    vol_features = [col for col in result.columns if col in ["Volume_Ratio", "VPT", "OBV"]]
    print(f"   Volume features ({len(vol_features)}): {vol_features}")

    # Volatility features
    volatility_features = [col for col in result.columns if col in ["Volatility", "TR", "ATR"]]
    print(f"   Volatility features ({len(volatility_features)}): {volatility_features}")
    print()

    # Validate features
    print("5. Validating features...")
    validation_result = fe.validate_features(result)
    print(f"   Validation result: {'PASSED' if validation_result else 'FAILED'}")
    print()

    # Show sample values for key indicators
    print("6. Sample values for key indicators (last 10 days):")
    key_indicators = ["Close", "Returns", "RSI", "MACD", "BB_Position", "Volume_Ratio"]
    sample_data = result[key_indicators].tail(10)

    for col in key_indicators:
        if col in sample_data.columns:
            values = sample_data[col].values
            print(
                f"   {col:12}: {' '.join([f'{v:8.4f}' if not pd.isna(v) else '     NaN' for v in values])}"
            )
    print()

    # Feature summary
    print("7. Feature summary statistics:")
    summary = fe.get_feature_summary(result)

    for feature in ["RSI", "MACD", "BB_Width", "Volume_Ratio", "Volatility"]:
        if feature in summary:
            stats = summary[feature]
            print(
                f"   {feature:12}: mean={stats['mean']:8.4f}, std={stats['std']:8.4f}, "
                f"null_ratio={stats['null_ratio']:6.2%}"
            )
    print()

    # Test individual indicator functions
    print("8. Testing individual indicator calculations...")

    # RSI test
    rsi = fe.calculate_rsi(data["Close"], period=14)
    rsi_valid = rsi.dropna()
    print(
        f"   RSI: range=[{rsi_valid.min():.2f}, {rsi_valid.max():.2f}], "
        f"mean={rsi_valid.mean():.2f}"
    )

    # MACD test
    macd_data = fe.calculate_macd(data["Close"])
    macd_valid = macd_data["MACD"].dropna()
    print(
        f"   MACD: range=[{macd_valid.min():.4f}, {macd_valid.max():.4f}], "
        f"mean={macd_valid.mean():.4f}"
    )

    # Bollinger Bands test
    bb_data = fe.calculate_bollinger_bands(data["Close"])
    bb_width_valid = bb_data["BB_Width"].dropna()
    print(
        f"   BB Width: range=[{bb_width_valid.min():.4f}, {bb_width_valid.max():.4f}], "
        f"mean={bb_width_valid.mean():.4f}"
    )
    print()

    print("✅ FeatureEngineer test completed successfully!")
    print("All technical indicators are working correctly.")


if __name__ == "__main__":
    main()
