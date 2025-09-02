import requests
import numpy as np
import json

def test_realistic_prediction():
    """Test prediction with realistic financial data"""
    
    print("=" * 60)
    print("TESTING WITH REALISTIC FINANCIAL DATA")
    print("=" * 60)
    
    # Load scaler to understand the expected input format
    with open("scalers/scaler_AAPL.json") as f:
        scaler = json.load(f)
    
    print(f"\nFeature dimensions expected: {len(scaler['feat_mean'])}")
    print(f"Features: {scaler['feature_names']}")
    
    # Generate realistic time series data
    np.random.seed(42)
    features = []
    
    # Start with a base price
    base_price = 175.0
    
    for day in range(60):
        # Simulate daily price movement
        daily_return = np.random.normal(0.001, 0.02)
        base_price *= (1 + daily_return)
        
        # OHLCV data
        open_price = base_price * np.random.uniform(0.99, 1.01)
        high = base_price * np.random.uniform(1.01, 1.03)
        low = base_price * np.random.uniform(0.97, 0.99)
        close = base_price
        volume = np.random.uniform(30_000_000, 70_000_000)
        
        # Technical indicators
        returns = daily_return
        volatility = abs(daily_return) * 2
        rsi = 50 + np.random.normal(0, 15)  # RSI oscillates around 50
        rsi = max(0, min(100, rsi))  # Bound between 0 and 100
        macd = np.random.normal(0, 0.5)
        bb_width = 2 + np.random.normal(0, 0.3)
        
        features.append([
            open_price, high, low, close, volume,
            returns, volatility, rsi, macd, bb_width
        ])
    
    # Make prediction request
    request_data = {
        "ticker": "AAPL",
        "features": features,
        "horizon": 3
    }
    
    print(f"\nSending request with {len(features)} timesteps")
    print(f"Last close price: ${features[-1][3]:.2f}")
    
    response = requests.post("http://localhost:8000/predict", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Prediction completed")
        
        if 'predictions' in result:
            preds = result['predictions']
            print(f"\nPredicted prices for next 3 days:")
            for i, pred in enumerate(preds[:3], 1):
                print(f"  Day +{i}: ${pred:.2f}")
        
        if 'confidence_intervals' in result:
            ci = result['confidence_intervals']
            print(f"\nConfidence intervals:")
            for i, (lower, upper) in enumerate(ci[:3], 1):
                print(f"  Day +{i}: ${lower:.2f} - ${upper:.2f}")
        
        if 'metadata' in result:
            print(f"\nMetadata: {result['metadata']}")
    else:
        print(f"\n[ERROR] Status: {response.status_code}")
        error_detail = response.json()
        print(f"Error: {error_detail}")
        
        # Detailed debugging
        if response.status_code == 500:
            print("\nDebugging info:")
            print("Check API terminal for full traceback")
            print("Common issues:")
            print("  1. Scaler fields mismatch")
            print("  2. Feature dimension mismatch")
            print("  3. Model expects different input shape")

if __name__ == "__main__":
    test_realistic_prediction()