import json
from pathlib import Path
import numpy as np

def create_proper_scalers():
    """Create scaler files with the correct structure the API expects"""
    
    print("=" * 60)
    print("CREATING PROPER SCALER FILES")
    print("=" * 60)
    
    scalers_dir = Path("scalers")
    scalers_dir.mkdir(exist_ok=True)
    
    tickers = ['AAPL', 'MSFT', 'GOOG']
    
    # Define realistic scaling parameters for financial data
    for ticker in tickers:
        print(f"\nCreating scaler for {ticker}...")
        
        # Get ticker-specific base prices for more realistic scaling
        base_prices = {
            'AAPL': 175.0,
            'MSFT': 380.0,
            'GOOG': 140.0
        }
        base_price = base_prices.get(ticker, 100.0)
        
        scaler_data = {
            # Feature scaling parameters (10 input features)
            "feat_mean": [
                base_price,           # open
                base_price * 1.01,    # high
                base_price * 0.99,    # low
                base_price,           # close
                50_000_000.0,         # volume
                0.0,                  # returns (centered at 0)
                0.02,                 # volatility
                50.0,                 # RSI
                0.0,                  # MACD
                2.0                   # Bollinger Band width
            ],
            "feat_std": [
                base_price * 0.1,     # open std
                base_price * 0.1,     # high std
                base_price * 0.1,     # low std
                base_price * 0.1,     # close std
                25_000_000.0,         # volume std
                0.02,                 # returns std
                0.01,                 # volatility std
                20.0,                 # RSI std
                1.0,                  # MACD std
                0.5                   # BB width std
            ],
            
            # Target scaling parameters (3 output values for 3-day predictions)
            "tgt_mean": [base_price, base_price, base_price],
            "tgt_std": [base_price * 0.05, base_price * 0.05, base_price * 0.05],
            
            # Additional info
            "feature_names": [
                "open", "high", "low", "close", "volume",
                "returns", "volatility", "rsi", "macd", "bb_width"
            ],
            "target_names": ["price_t+1", "price_t+2", "price_t+3"],
            "scaler_type": "StandardScaler",
            "ticker": ticker,
            
            # Keep the original fields for compatibility
            "mean": [
                base_price, base_price * 1.01, base_price * 0.99, base_price,
                50_000_000.0, 0.0, 0.02, 50.0, 0.0, 2.0
            ],
            "std": [
                base_price * 0.1, base_price * 0.1, base_price * 0.1, base_price * 0.1,
                25_000_000.0, 0.02, 0.01, 20.0, 1.0, 0.5
            ]
        }
        
        scaler_path = scalers_dir / f"scaler_{ticker}.json"
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        print(f"  [OK] Created {scaler_path.name}")
        
        # Verify the structure
        print(f"  Features: {len(scaler_data['feat_mean'])} dimensions")
        print(f"  Targets: {len(scaler_data['tgt_mean'])} dimensions")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Verify all scalers have the required fields
    for ticker in tickers:
        scaler_path = scalers_dir / f"scaler_{ticker}.json"
        with open(scaler_path) as f:
            data = json.load(f)
        
        required_fields = ['feat_mean', 'feat_std', 'tgt_mean', 'tgt_std']
        missing = [field for field in required_fields if field not in data]
        
        if missing:
            print(f"[ERROR] {ticker}: Missing fields {missing}")
        else:
            print(f"[OK] {ticker}: All required fields present")
    
    print("\n[SUCCESS] Proper scaler files created")
    return True

if __name__ == "__main__":
    create_proper_scalers()