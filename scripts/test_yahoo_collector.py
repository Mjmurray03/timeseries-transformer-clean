#!/usr/bin/env python3
"""
Simple test script to demonstrate YahooFinanceCollector functionality.
This script shows how to use the collector to download stock data.
"""

import asyncio
import sys
import os
from datetime import date

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config.data_config import DataConfig
from src.data.collectors.yahoo_finance import YahooFinanceCollector, download_ticker


async def main():
    """Demonstrate YahooFinanceCollector usage."""
    
    # Create a simple configuration for testing
    config_dict = {
        "data_sources": {
            "yahoo_finance": {
                "enabled": True,
                "rate_limit": 2,  # Conservative rate limit
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1
            }
        },
        "data_quality": {
            "min_trading_days": 10,  # Reduced for demo
            "max_missing_ratio": 0.1,
            "outlier_threshold": 10.0,
            "min_volume": 10000,  # Reduced for demo
            "min_price": 1.0,
            "max_price": 10000.0
        },
        "tickers": {
            "demo": ["AAPL", "MSFT"]
        },
        "date_ranges": {
            "recent": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            }
        }
    }
    
    config = DataConfig(config_dict)
    
    print("🚀 Testing YahooFinanceCollector")
    print("=" * 50)
    
    # Test 1: Initialize collector
    print("\n1. Initializing collector...")
    try:
        collector = YahooFinanceCollector(config)
        print("✅ Collector initialized successfully")
        print(f"   Rate limit: {collector.source_config.rate_limit}/sec")
        print(f"   Timeout: {collector.source_config.timeout}s")
    except Exception as e:
        print(f"❌ Failed to initialize collector: {e}")
        return
    
    # Test 2: Download single ticker
    print("\n2. Downloading single ticker (AAPL)...")
    try:
        data = await collector.collect_ticker(
            "AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        if data is not None:
            print("✅ Successfully downloaded AAPL data")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"   Sample data:")
            print(data.head(3).to_string())
        else:
            print("❌ Failed to download AAPL data")
    except Exception as e:
        print(f"❌ Error downloading AAPL: {e}")
    
    # Test 3: Download multiple tickers
    print("\n3. Downloading multiple tickers...")
    try:
        tickers = ["AAPL", "MSFT"]
        data_dict = await collector.collect_multiple(
            tickers,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            max_concurrent=2
        )
        
        print(f"✅ Downloaded data for {len(data_dict)} tickers")
        for ticker, data in data_dict.items():
            print(f"   {ticker}: {data.shape[0]} days, {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error downloading multiple tickers: {e}")
    
    # Test 4: Show statistics
    print("\n4. Collection statistics:")
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 5: Test convenience function
    print("\n5. Testing convenience function...")
    try:
        data = await download_ticker("AAPL", config, "recent")
        if data is not None:
            print("✅ Convenience function works")
            print(f"   Downloaded {len(data)} days of data")
        else:
            print("❌ Convenience function returned None")
    except Exception as e:
        print(f"❌ Convenience function error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 YahooFinanceCollector test completed!")


if __name__ == "__main__":
    asyncio.run(main())