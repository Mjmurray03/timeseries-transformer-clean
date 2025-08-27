# tests/integration/test_data_pipeline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.collectors.yahoo_finance import YahooFinanceCollector
from src.data.validators import DataValidator
from src.data.processors.feature_engineering import FeatureEngineer
from src.data.storage import DataStorage
from src.config.data_config import DataConfig
import pandas as pd
import yaml
from datetime import datetime, timedelta
import asyncio

def test_complete_data_pipeline():
    """Test the entire data pipeline with Doppler integration."""
    
    print("[*] Testing Complete Data Pipeline...")
    
    # Load configuration from YAML
    config_path = Path(__file__).parent.parent.parent / "configs" / "data_config.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = DataConfig(config_dict)
    
    # Test 1: Data Collection with minimal data
    print("\n[1] Testing YahooFinanceCollector...")
    collector = YahooFinanceCollector(config)
    
    # Download 13 months for proper validation (quick for testing with just 2 tickers)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    test_tickers = ['AAPL', 'MSFT']
    
    # Run async collection
    async def collect_data():
        data = {}
        for ticker in test_tickers:
            ticker_data = await collector.collect_ticker(
                ticker=ticker,
                start_date=start_date.date(),
                end_date=end_date.date()
            )
            if ticker_data is not None:
                data[ticker] = ticker_data
        return data
    
    # Run the async function
    data = asyncio.run(collect_data())
    
    assert len(data) > 0, "[X] No data collected"
    assert all(ticker in data for ticker in test_tickers), "[X] Missing tickers"
    print(f"[OK] Collected data for {len(data)} tickers")
    if 'AAPL' in data:
        print(f"   Sample shape: {data['AAPL'].shape}")
    
    # Test 2: Data Validation
    print("\n[2] Testing DataValidator...")
    validator = DataValidator(config)
    
    for ticker, df in data.items():
        validation_result = validator.validate(df, ticker)
        is_valid = validation_result.is_valid
        report = [str(issue) for issue in validation_result.issues]
        assert is_valid, f"[X] Validation failed for {ticker}: {report}"
        print(f"[OK] {ticker} validation passed")
    
    # Test 3: Feature Engineering
    print("\n[3] Testing FeatureEngineer...")
    engineer = FeatureEngineer()
    
    engineered_data = {}
    for ticker, df in data.items():
        engineered_df = engineer.engineer_features(df)
        assert len(engineered_df.columns) > len(df.columns), "[X] No features added"
        engineered_data[ticker] = engineered_df
        
        # Check specific indicators exist
        expected_features = ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
        for feature in expected_features:
            assert feature in engineered_df.columns, f"[X] Missing {feature}"
        
        print(f"[OK] {ticker}: Added {len(engineered_df.columns) - len(df.columns)} features")
    
    # Test 4: Data Storage
    print("\n[4] Testing DataStorage...")
    storage = DataStorage()
    
    for ticker, df in engineered_data.items():
        # Save processed data
        file_path = storage.save_processed_data(df, ticker)
        
        # Load it back
        loaded_df, _ = storage.load_processed_data(ticker)
        assert loaded_df is not None, f"[X] Failed to load {ticker}"
        assert len(loaded_df) == len(df), f"[X] Data mismatch for {ticker}"
        print(f"[OK] {ticker}: Saved and loaded {len(loaded_df)} rows")
    
    print("\n[SUCCESS] Complete data pipeline test PASSED!")
    return True

# Run with Doppler
if __name__ == "__main__":
    test_complete_data_pipeline()