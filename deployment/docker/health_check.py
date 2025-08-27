#!/usr/bin/env python3
"""Health check script for container health monitoring"""

import sys
import requests
import time
from typing import Dict, Any

def check_health() -> bool:
    """Check if the service is healthy"""
    try:
        # Check API health endpoint
        response = requests.get(
            "http://localhost:8000/health",
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
        
        health_data = response.json()
        
        # Verify required health indicators
        required_checks = ["status", "model_loaded", "api_ready"]
        for check in required_checks:
            if check not in health_data:
                print(f"Missing health indicator: {check}")
                return False
            
            if check == "status" and health_data[check] != "healthy":
                print(f"Service status: {health_data[check]}")
                return False
            
            if check in ["model_loaded", "api_ready"] and not health_data[check]:
                print(f"{check} check failed")
                return False
        
        # Check model inference capability
        if "inference_ready" in health_data and not health_data["inference_ready"]:
            print("Model inference not ready")
            return False
        
        print("Health check passed")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in health check: {e}")
        return False

def check_readiness() -> bool:
    """Check if the service is ready to accept traffic"""
    try:
        response = requests.get(
            "http://localhost:8000/ready",
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"Readiness check failed: HTTP {response.status_code}")
            return False
        
        ready_data = response.json()
        
        # Check all readiness conditions
        if not ready_data.get("ready", False):
            print("Service not ready")
            return False
        
        # Verify model is warmed up
        if ready_data.get("warmup_complete", False):
            print("Readiness check passed")
            return True
        else:
            print("Model warmup incomplete")
            return False
            
    except Exception as e:
        print(f"Readiness check failed: {e}")
        return False

if __name__ == "__main__":
    # Determine check type from command line argument
    check_type = sys.argv[1] if len(sys.argv) > 1 else "health"
    
    if check_type == "ready":
        success = check_readiness()
    else:
        success = check_health()
    
    sys.exit(0 if success else 1)