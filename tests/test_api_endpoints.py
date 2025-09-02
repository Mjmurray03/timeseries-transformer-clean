import requests
import json
import numpy as np
from pathlib import Path
import sys
import time

def test_api_endpoints():
    """Test all API endpoints."""
    
    BASE_URL = "http://localhost:8000"
    
    print("=" * 60)
    print("API ENDPOINT TESTS")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n[TEST 1] Health Check Endpoint")
    print("-" * 40)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
        else:
            print(f"[FAIL] Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API. Is the server running?")
        print("Start the API with: python -m uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False
    
    # Test 2: Model Info Endpoint
    print("\n[TEST 2] Model Info Endpoint")
    print("-" * 40)
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Status Code: {response.status_code}")
            data = response.json()
            print(f"Model Version: {data.get('version', 'N/A')}")
            print(f"Model Type: {data.get('model_type', 'N/A')}")
        else:
            print(f"[WARNING] Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"[WARNING] Model info endpoint not available: {e}")
    
    # Test 3: Prediction Endpoint
    print("\n[TEST 3] Prediction Endpoint")
    print("-" * 40)
    
    # Create sample input data (60 timesteps, 8 features)
    sample_data = {
        "ticker": "AAPL",
        "features": np.random.randn(60, 8).tolist(),  # 60 days, 8 features
        "horizon": 3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"[OK] Status Code: {response.status_code}")
            result = response.json()
            print(f"Prediction Result Keys: {list(result.keys())}")
            if 'predictions' in result:
                preds = result['predictions']
                print(f"Predictions Shape: {len(preds)} values")
                print(f"Sample Predictions: {preds[:3] if len(preds) >= 3 else preds}")
        else:
            print(f"[FAIL] Status Code: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
    
    # Test 4: Batch Prediction (if available)
    print("\n[TEST 4] Batch Prediction Endpoint")
    print("-" * 40)
    
    batch_data = {
        "requests": [
            {
                "ticker": "AAPL",
                "features": np.random.randn(60, 8).tolist(),
                "horizon": 3
            },
            {
                "ticker": "MSFT", 
                "features": np.random.randn(60, 8).tolist(),
                "horizon": 3
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json=batch_data,
            timeout=15
        )
        
        if response.status_code == 200:
            print(f"[OK] Status Code: {response.status_code}")
            result = response.json()
            print(f"Batch Results: {len(result.get('results', []))} predictions")
        else:
            print(f"[INFO] Batch endpoint not available or different format")
            print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"[INFO] Batch prediction not available: {e}")
    
    # Test 5: Check for Swagger/Docs
    print("\n[TEST 5] API Documentation")
    print("-" * 40)
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Swagger docs available at: {BASE_URL}/docs")
        else:
            print(f"[INFO] Swagger docs not available")
    except:
        print(f"[INFO] No automatic API documentation")
    
    print("\n" + "=" * 60)
    print("API TESTS COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Wait a moment for server to be ready if just started
    print("Starting API tests in 2 seconds...")
    time.sleep(2)
    
    success = test_api_endpoints()
    sys.exit(0 if success else 1)