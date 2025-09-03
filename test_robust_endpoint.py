#!/usr/bin/env python
"""
Test script for the robust prediction endpoint
"""
import requests
import json
import random
import numpy as np

# API endpoint
BASE_URL = "http://localhost:8000"

def test_flat_list_format():
    """Test the endpoint with flat list format (600 elements)"""
    print("="*60)
    print("Testing Flat List Format (600 elements)")
    print("="*60)
    
    # Generate 600 random features (60 days * 10 features per day)
    features = [round(random.uniform(0.1, 1.0), 4) for _ in range(600)]
    
    payload = {
        "ticker": "AAPL",
        "features": features
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("[SUCCESS] Flat list format accepted!")
            print(f"Prediction keys: {list(result.keys())}")
            if 'predictions' in result:
                pred_keys = list(result['predictions'].keys())
                print(f"Prediction types: {pred_keys}")
            print(f"Input format processed: 600 elements -> reshaped to (60, 10)")
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def test_2d_array_format():
    """Test the endpoint with 2D array format (60x10)"""
    print("\n" + "="*60)
    print("Testing 2D Array Format (60x10)")
    print("="*60)
    
    # Generate 60x10 array (60 days, 10 features each)
    features_2d = []
    for day in range(60):
        day_features = [round(random.uniform(0.1, 1.0), 4) for _ in range(10)]
        features_2d.append(day_features)
    
    payload = {
        "ticker": "AAPL", 
        "features": features_2d
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("[SUCCESS] 2D array format accepted!")
            print(f"Prediction keys: {list(result.keys())}")
            if 'predictions' in result:
                pred_keys = list(result['predictions'].keys())
                print(f"Prediction types: {pred_keys}")
            print(f"Input format processed: (60, 10) array")
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error response: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    test_cases = [
        # Wrong number of elements for flat list
        {
            "name": "Wrong flat list size (500 instead of 600)",
            "payload": {
                "ticker": "AAPL",
                "features": [0.5] * 500
            }
        },
        # Wrong shape for 2D array
        {
            "name": "Wrong 2D array shape (50x10 instead of 60x10)",
            "payload": {
                "ticker": "AAPL", 
                "features": [[0.5] * 10 for _ in range(50)]
            }
        },
        # Invalid ticker
        {
            "name": "Invalid ticker",
            "payload": {
                "ticker": "INVALID_TICKER",
                "features": [0.5] * 600
            }
        },
        # NaN values
        {
            "name": "NaN values in features",
            "payload": {
                "ticker": "AAPL",
                "features": [float('nan')] * 600
            }
        },
        # Mixed data types (strings in numeric array)
        {
            "name": "Invalid data types (strings in numeric array)",
            "payload": {
                "ticker": "AAPL",
                "features": ["invalid"] * 600
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(f"{BASE_URL}/predict", json=test_case['payload'], timeout=30)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 422:
                try:
                    error_detail = response.json()
                    print("[SUCCESS] Proper validation error returned")
                    if 'detail' in error_detail:
                        print(f"Error message: {error_detail['detail']}")
                except:
                    print(f"Error response: {response.text}")
            elif response.status_code == 400:
                try:
                    error_detail = response.json()
                    print("[SUCCESS] Proper client error returned")
                    print(f"Error message: {error_detail.get('detail', 'No detail')}")
                except:
                    print(f"Error response: {response.text}")
            else:
                print(f"[UNEXPECTED] Status code {response.status_code}")
                try:
                    print(f"Response: {response.json()}")
                except:
                    print(f"Response: {response.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request exception: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")

def test_api_health():
    """Test API health endpoint first"""
    print("="*60)
    print("Testing API Health")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("[SUCCESS] API is healthy")
            print(f"Available models: {health_data.get('models', 'Not available')}")
        else:
            print(f"[ERROR] Health check failed")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Cannot connect to API: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Robust Prediction Endpoint Test Suite")
    print("=" * 60)
    
    # Test API health first
    if not test_api_health():
        print("API is not accessible. Make sure the server is running on localhost:8000")
        exit(1)
    
    # Run all tests
    test_flat_list_format()
    test_2d_array_format()
    test_error_handling()
    
    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)