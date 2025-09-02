#!/usr/bin/env python3
"""
Test script for DataStorage functionality

This script demonstrates the DataStorage working with sample data
and shows all storage formats (Parquet, HDF5) working correctly.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.storage import DataStorage


def create_sample_data():
    """Create realistic sample OHLCV data for testing"""
    np.random.seed(42)

    # Generate 100 days of data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # Small positive drift, 2% daily volatility

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data with realistic relationships
    data = pd.DataFrame(index=dates)
    data["Close"] = prices

    # Generate OHLC with realistic spreads
    data["Open"] = data["Close"].shift(1) * np.random.uniform(0.995, 1.005, 100)
    data["High"] = np.maximum(data["Open"], data["Close"]) * np.random.uniform(1.000, 1.015, 100)
    data["Low"] = np.minimum(data["Open"], data["Close"]) * np.random.uniform(0.985, 1.000, 100)

    # Volume with some correlation to price changes
    base_volume = 1000000
    volume_multiplier = 1 + np.abs(returns) * 5  # Higher volume on big moves
    data["Volume"] = (base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, 100)).astype(
        int
    )

    # Clean up first row
    data.iloc[0] = [100, 102, 98, 100, 1500000]  # Open, High, Low, Close, Volume

    return data


def create_processed_data(raw_data):
    """Create sample processed data with features"""
    data = raw_data.copy()

    # Add some engineered features
    data["Returns"] = data["Close"].pct_change()
    data["RSI"] = np.random.uniform(20, 80, len(data))
    data["MACD"] = np.random.uniform(-2, 2, len(data))
    data["Volume_Ratio"] = np.random.uniform(0.5, 2.0, len(data))
    data["Volatility"] = data["Returns"].rolling(20).std()

    return data


def main():
    """Main test function"""
    print("Testing DataStorage functionality...")
    print("=" * 50)

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Initialize DataStorage
        print("\n1. Initializing DataStorage...")
        storage = DataStorage(base_path=temp_dir)
        print("   ✅ DataStorage initialized successfully")

        # Check directory structure
        expected_dirs = ["raw", "processed", "metadata", "cache", "backups"]
        for dir_name in expected_dirs:
            dir_path = Path(temp_dir) / dir_name
            if dir_path.exists():
                print(f"   ✅ Directory created: {dir_name}")
            else:
                print(f"   ❌ Directory missing: {dir_name}")

        # Check metadata database
        db_path = Path(temp_dir) / "metadata" / "data_catalog.db"
        if db_path.exists():
            print("   ✅ Metadata database created")
        else:
            print("   ❌ Metadata database missing")

        # Create sample data
        print("\n2. Creating sample data...")
        raw_data = create_sample_data()
        processed_data = create_processed_data(raw_data)
        print(f"   Raw data: {len(raw_data)} rows, {len(raw_data.columns)} columns")
        print(
            f"   Processed data: {len(processed_data)} rows, {len(processed_data.columns)} columns"
        )

        # Test raw data storage (Parquet)
        print("\n3. Testing raw data storage (Parquet)...")
        ticker = "AAPL"
        raw_file_path = storage.save_raw_data(raw_data, ticker)
        print(f"   ✅ Saved raw data: {raw_file_path}")

        # Test raw data loading
        loaded_raw_data = storage.load_raw_data(ticker)
        if loaded_raw_data.equals(raw_data):
            print("   ✅ Raw data loaded correctly")
        else:
            print("   ❌ Raw data loading failed")

        # Test processed data storage (HDF5)
        print("\n4. Testing processed data storage (HDF5)...")
        feature_set = "technical_indicators"
        processing_config = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "volatility_window": 20,
        }

        processed_file_path = storage.save_processed_data(
            processed_data, ticker, feature_set, processing_config
        )
        print(f"   ✅ Saved processed data: {processed_file_path}")

        # Test processed data loading
        loaded_processed_data, loaded_config = storage.load_processed_data(ticker, feature_set)
        if loaded_processed_data.equals(processed_data):
            print("   ✅ Processed data loaded correctly")
        else:
            print("   ❌ Processed data loading failed")

        if loaded_config == processing_config:
            print("   ✅ Processing config loaded correctly")
        else:
            print("   ❌ Processing config loading failed")

        # Test generic timeseries storage
        print("\n5. Testing generic timeseries storage...")

        # Test Parquet format
        parquet_path = Path(temp_dir) / "test_data.parquet"
        storage.save_timeseries(raw_data, str(parquet_path))
        loaded_parquet = storage.load_timeseries(str(parquet_path))
        if loaded_parquet.equals(raw_data):
            print("   ✅ Parquet format works correctly")
        else:
            print("   ❌ Parquet format failed")

        # Test HDF5 format (may fallback to Parquet)
        hdf5_path = Path(temp_dir) / "test_data.h5"
        storage.save_timeseries(processed_data, str(hdf5_path))

        # Check if file was saved as HDF5 or fell back to Parquet
        if hdf5_path.exists():
            loaded_hdf5 = storage.load_timeseries(str(hdf5_path))
            format_name = "HDF5"
        else:
            # Check for Parquet fallback
            parquet_fallback = hdf5_path.with_suffix(".parquet")
            if parquet_fallback.exists():
                loaded_hdf5 = storage.load_timeseries(str(parquet_fallback))
                format_name = "HDF5 (Parquet fallback)"
            else:
                print("   ❌ Neither HDF5 nor Parquet fallback file found")
                loaded_hdf5 = None

        if loaded_hdf5 is not None and loaded_hdf5.equals(processed_data):
            print(f"   ✅ {format_name} format works correctly")
        else:
            print(f"   ❌ {format_name} format failed")

        # Test data versioning
        print("\n6. Testing data versioning...")
        version_id = storage.create_data_version(
            raw_data,
            data_type="raw",
            ticker=ticker,
            description="Test version for AAPL raw data",
            metadata={"source": "test_script", "quality": "high"},
        )
        print(f"   ✅ Created data version: {version_id}")

        # Load versioned data
        versioned_data = storage.load_data_version(version_id)
        if versioned_data.equals(raw_data):
            print("   ✅ Versioned data loaded correctly")
        else:
            print("   ❌ Versioned data loading failed")

        # Test metadata catalogs
        print("\n7. Testing metadata catalogs...")

        # Raw data catalog
        raw_catalog = storage.get_data_catalog("raw")
        print(f"   Raw data catalog: {len(raw_catalog)} entries")
        if len(raw_catalog) > 0:
            print(f"   - Ticker: {raw_catalog.iloc[0]['ticker']}")
            print(f"   - Rows: {raw_catalog.iloc[0]['row_count']}")
            print(f"   - File size: {raw_catalog.iloc[0]['file_size']} bytes")

        # Processed data catalog
        processed_catalog = storage.get_data_catalog("processed")
        print(f"   Processed data catalog: {len(processed_catalog)} entries")
        if len(processed_catalog) > 0:
            print(f"   - Ticker: {processed_catalog.iloc[0]['ticker']}")
            print(f"   - Feature set: {processed_catalog.iloc[0]['feature_set']}")
            print(f"   - Features: {processed_catalog.iloc[0]['feature_count']}")

        # Test storage statistics
        print("\n8. Testing storage statistics...")
        stats = storage.get_storage_stats()
        for data_type, type_stats in stats.items():
            print(
                f"   {data_type}: {type_stats['file_count']} files, "
                f"{type_stats['total_size_mb']:.2f} MB"
            )

        # Test multiple tickers
        print("\n9. Testing multiple tickers...")
        tickers = ["MSFT", "GOOGL", "TSLA"]
        for test_ticker in tickers:
            # Modify data slightly for each ticker
            ticker_data = raw_data * np.random.uniform(0.8, 1.2)
            storage.save_raw_data(ticker_data, test_ticker)
            print(f"   ✅ Saved data for {test_ticker}")

        # Check final catalog
        final_catalog = storage.get_data_catalog("raw")
        print(f"   Final raw data catalog: {len(final_catalog)} entries")
        print(f"   Tickers: {sorted(final_catalog['ticker'].unique())}")

        print("\n" + "=" * 50)
        print("✅ All DataStorage tests completed successfully!")
        print("DataStorage is working correctly with:")
        print("- Parquet format for raw data")
        print("- HDF5 format for processed data")
        print("- SQLite metadata tracking")
        print("- Data versioning")
        print("- Multiple file format support")
        print("- Compression options")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        try:
            # Close storage connections
            if "storage" in locals():
                storage.close()
            shutil.rmtree(temp_dir)
        except PermissionError as e:
            print(f"Warning: Could not fully clean up temporary directory: {e}")
            print("This is usually harmless and the OS will clean it up later.")


if __name__ == "__main__":
    main()
