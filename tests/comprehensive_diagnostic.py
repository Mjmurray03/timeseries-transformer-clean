import json
import torch
import sys
import requests
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))

def run_comprehensive_diagnostic():
    """Diagnose ALL system components and identify mismatches"""
    
    print("=" * 60)
    print("COMPREHENSIVE SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    issues = []
    
    # 1. Check Model Configuration
    print("\n[1] MODEL CONFIGURATION")
    print("-" * 40)
    
    try:
        from src.models.timeseries_transformer import TimeSeriesTransformer
        
        # Create a model to check its expected input
        model = TimeSeriesTransformer(
            input_dim=10,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            max_seq_length=60,
            output_dim=3,
            forecast_horizon=5,
            use_attention_pooling=True
        )
        
        print(f"Model expects input_dim: 10")
        print(f"Model expects seq_length: 60")
        print(f"Model output_dim: 3")
        
        # Test with correct shape
        test_input = torch.randn(1, 60, 10)
        output = model(test_input)
        print(f"Model forward pass successful with shape (1, 60, 10)")
        
    except Exception as e:
        issues.append(f"Model configuration issue: {e}")
        print(f"ERROR: {e}")
    
    # 2. Check Scaler Configuration
    print("\n[2] SCALER CONFIGURATION")
    print("-" * 40)
    
    scaler_path = Path("scalers/scaler_AAPL.json")
    if scaler_path.exists():
        with open(scaler_path) as f:
            scaler = json.load(f)
        
        print(f"Scaler feat_mean length: {len(scaler.get('feat_mean', []))}")
        print(f"Scaler feat_std length: {len(scaler.get('feat_std', []))}")
        print(f"Scaler feature_names: {scaler.get('feature_names', [])}")
        
        if len(scaler.get('feat_mean', [])) != 10:
            issues.append(f"Scaler has {len(scaler.get('feat_mean', []))} features, model expects 10")
    else:
        issues.append("Scaler file not found")
    
    # 3. Check API Endpoint Expectations
    print("\n[3] API ENDPOINT ANALYSIS")
    print("-" * 40)
    
    # Read the API code to understand what it expects
    api_file = Path("src/api/main.py")
    if api_file.exists():
        with open(api_file) as f:
            api_content = f.read()
        
        # Check for validation logic
        if "must have exactly 8 features" in api_content:
            print("API validates for 8 features (MISMATCH with model's 10)")
            issues.append("API expects 8 features but model needs 10")
        
        # 5. Find the Validation Code
        print("\n[5] VALIDATION CODE LOCATION")
        print("-" * 40)
        
        # Search for where the 8 features validation happens
        for i, line in enumerate(api_content.split('\n'), 1):
            if "8 features" in line or ("8" in line and "features" in line):
                print(f"Line {i}: {line.strip()}")
    else:
        issues.append("API main.py file not found")
        print("API main.py file not found")
    
    # 4. Check Actual API Behavior
    print("\n[4] LIVE API TEST")
    print("-" * 40)
    
    try:
        # Test with 8 features (what API validation might expect)
        test_8_features = {
            "ticker": "AAPL",
            "features": [[1.0] * 8 for _ in range(60)],
            "horizon": 3
        }
        
        response = requests.post("http://localhost:8000/predict", json=test_8_features, timeout=5)
        print(f"8 features test: Status {response.status_code}")
        if response.status_code != 200:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"  Error: {error_detail}")
            except:
                print(f"  Error: {response.text}")
            
        # Test with 10 features (what model expects)
        test_10_features = {
            "ticker": "AAPL",
            "features": [[1.0] * 10 for _ in range(60)],
            "horizon": 3
        }
        
        response = requests.post("http://localhost:8000/predict", json=test_10_features, timeout=5)
        print(f"10 features test: Status {response.status_code}")
        if response.status_code != 200:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"  Error: {error_detail}")
            except:
                print(f"  Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        issues.append("API not running")
        print("API not accessible")
    except Exception as e:
        issues.append(f"API test error: {e}")
        print(f"API test error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nROOT CAUSE:")
        print("The API has hardcoded validation for 8 features,")
        print("but the model and scalers are configured for 10 features.")
        
        print("\nSOLUTION:")
        print("Need to update API validation to accept 10 features,")
        print("or reconfigure everything to use 8 features.")
    else:
        print("No major issues found")
    
    return issues

if __name__ == "__main__":
    issues = run_comprehensive_diagnostic()