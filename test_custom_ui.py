#!/usr/bin/env python
"""
Test script for the custom Swagger UI implementation
"""
import requests
import time
import sys
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:8000"

def test_custom_docs_endpoint():
    """Test that custom /docs endpoint is accessible and returns HTML"""
    print("=" * 60)
    print("Testing Custom Swagger UI Documentation")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            
            # Check for key elements in the custom HTML
            checks = [
                ("HTML structure", "<!DOCTYPE html>" in content),
                ("Custom title", "TimeSeries Transformer API - Interactive Documentation" in content),
                ("Dark theme CSS", "background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%)" in content),
                ("Educational content", "Student Guide: Understanding the API" in content),
                ("Quick start guide", "Quick Start" in content),
                ("Custom header", "🚀 TimeSeries Transformer API" in content),
                ("Swagger UI integration", "swagger-ui-bundle.js" in content),
                ("Educational tooltips", "What Each Endpoint Does:" in content),
            ]
            
            print("[SUCCESS] Custom docs endpoint is accessible!")
            print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
            print(f"Content length: {len(content):,} characters")
            
            print("\nContent validation:")
            all_passed = True
            for check_name, passed in checks:
                status = "[OK]" if passed else "[FAIL]"
                print(f"  {status} {check_name}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print("\n[SUCCESS] All custom UI elements are present!")
                return True
            else:
                print("\n[WARNING] Some custom UI elements are missing")
                return False
        else:
            print(f"[ERROR] Request failed with status {response.status_code}")
            try:
                print(f"Error response: {response.text}")
            except:
                print("Could not decode error response")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_openapi_json_endpoint():
    """Test that OpenAPI JSON endpoint is still accessible for Swagger UI"""
    print("\n" + "=" * 60)
    print("Testing OpenAPI JSON Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            json_data = response.json()
            print("[SUCCESS] OpenAPI JSON is accessible!")
            print(f"API Title: {json_data.get('info', {}).get('title', 'Not found')}")
            print(f"API Version: {json_data.get('info', {}).get('version', 'Not found')}")
            print(f"Number of paths: {len(json_data.get('paths', {}))}")
            
            # Check for educational descriptions in endpoints
            paths = json_data.get('paths', {})
            educational_endpoints = []
            for path, methods in paths.items():
                for method, details in methods.items():
                    summary = details.get('summary', '')
                    if any(emoji in summary for emoji in ['🏥', '🎯', '📈', '🧠']):
                        educational_endpoints.append(f"{method.upper()} {path}")
            
            if educational_endpoints:
                print(f"[SUCCESS] Found {len(educational_endpoints)} educational endpoints:")
                for endpoint in educational_endpoints:
                    print(f"  - {endpoint}")
            else:
                print("[WARNING] No educational endpoint descriptions found")
            
            return True
        else:
            print(f"[ERROR] OpenAPI JSON request failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request exception: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_api_health():
    """Test API health endpoint first"""
    print("=" * 60)
    print("Testing API Health")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("[SUCCESS] API is healthy")
            print(f"Available models: {health_data.get('models_loaded', 'Not available')}")
            print(f"CUDA available: {health_data.get('cuda_available', 'Unknown')}")
            return True
        else:
            print(f"[ERROR] Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Cannot connect to API: {e}")
        return False

def check_html_file_exists():
    """Check if custom_docs.html file exists"""
    print("=" * 60)
    print("Checking HTML File")
    print("=" * 60)
    
    html_file = Path("src/api/custom_docs.html")
    if html_file.exists():
        print(f"[SUCCESS] Custom HTML file exists: {html_file}")
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"File size: {len(content):,} characters")
        return True
    else:
        print(f"[ERROR] Custom HTML file not found: {html_file}")
        return False

def display_instructions():
    """Display instructions for manual testing"""
    print("\n" + "=" * 60)
    print("Manual Testing Instructions")
    print("=" * 60)
    print("1. Open your web browser")
    print("2. Navigate to: http://localhost:8000/docs")
    print("3. Verify you see:")
    print("   - Dark theme with gradient backgrounds")
    print("   - Custom header with rocket emoji")
    print("   - Educational content sections")
    print("   - Quick start guide on the right")
    print("   - Interactive Swagger UI with endpoints")
    print("   - Educational descriptions on endpoints")
    print("4. Try interacting with an endpoint (like /health)")
    print("5. Verify the interface is responsive and functional")

if __name__ == "__main__":
    print("Custom Swagger UI Verification Test Suite")
    print("=" * 60)
    
    # Check if HTML file exists
    html_exists = check_html_file_exists()
    if not html_exists:
        print("\n[CRITICAL] HTML file missing. Cannot proceed with API tests.")
        sys.exit(1)
    
    # Test API health first
    if not test_api_health():
        print("\n[WARNING] API is not accessible. Make sure the server is running.")
        print("Start the server with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)
    
    # Run tests
    docs_test = test_custom_docs_endpoint()
    json_test = test_openapi_json_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    results = [
        ("HTML file exists", html_exists),
        ("API health check", True),  # We already passed this
        ("Custom /docs endpoint", docs_test),
        ("OpenAPI JSON endpoint", json_test),
    ]
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    if all_passed:
        print("\n🎉 All tests passed! Custom Swagger UI is working correctly!")
        display_instructions()
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        
    print("\n" + "=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)