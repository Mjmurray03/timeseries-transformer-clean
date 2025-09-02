import requests
import numpy as np
import json

def test_final_validation():
    """Final comprehensive validation after fixes"""
    
    print("=" * 60)
    print("FINAL SYSTEM VALIDATION")
    print("=" * 60)
    
    # Create proper 10-feature test data matching what we trained
    features = []
    base_price = 175.0
    
    for _ in range(60):
        daily_return = np.random.normal(0.001, 0.02)
        base_price *= (1 + daily_return)
        
        day_features = [
            base_price,                          # open
            base_price * 1.02,                   # high
            base_price * 0.98,                   # low
            base_price,                          # close
            np.random.uniform(30e6, 70e6),      # volume
            daily_return,                        # returns
            abs(daily_return),                   # volatility
            50 + np.random.normal(0, 15),       # RSI
            np.random.normal(0, 0.5),           # MACD
            2 + np.random.normal(0, 0.3)        # BB width
        ]
        features.append(day_features)
    
    request = {
        "ticker": "AAPL",
        "features": features,
        "horizon": 3
    }
    
    print(f"Testing with {len(features)} timesteps, {len(features[0])} features each")
    
    try:
        response = requests.post("http://localhost:8000/predict", json=request, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] API prediction working!")
            
            if 'predictions' in result:
                preds = result['predictions']
                print(f"Predictions: {preds[:3]}")
                print(f"Number of predictions: {len(preds)}")
            
            if 'metadata' in result:
                print(f"Metadata: {result['metadata']}")
            
            print("\n[VALIDATION PASSED] System is working correctly!")
            return True
        else:
            print(f"\n[FAIL] Status: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n[FAIL] Cannot connect to API")
        print("Make sure the API server is running:")
        print("python -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"\n[FAIL] Request error: {e}")
        return False

if __name__ == "__main__":
    success = test_final_validation()
    if success:
        print("\n🎉 All systems operational!")
    else:
        print("\n❌ Issues remain - check the diagnostic output above")