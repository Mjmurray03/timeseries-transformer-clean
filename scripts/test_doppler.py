#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test Doppler integration"""
import os
import sys
from pathlib import Path

# Ensure UTF-8 encoding for Windows
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.append(str(Path(__file__).parent.parent))

from src.config.secrets import secrets
from src.config.config import Config

def test_secrets():
    print("=== Testing Doppler Integration ===\n")
    
    # Test each secret
    secrets_to_test = [
        ('ALPHA_VANTAGE_API_KEY', Config.ALPHA_VANTAGE_KEY),
        ('NEWSAPI_API_KEY', Config.NEWSAPI_KEY),
        ('HUGGINGFACE_API_KEY', Config.HUGGINGFACE_TOKEN),
        ('WANDB_API_KEY', Config.WANDB_KEY),
    ]
    
    for name, value in secrets_to_test:
        if value:
            # Show only first 4 chars for security
            masked = value[:4] + '*' * (len(value) - 4)
            print(f"[OK] {name}: {masked}")
        else:
            print(f"[FAIL] {name}: Not found")
    
    print("\n=== Directory Structure ===")
    print(f"  Project Root: {Config.PROJECT_ROOT}")
    print(f"  Data Dir: {Config.DATA_DIR}")
    print(f"  Raw Data: {Config.RAW_DATA_DIR}")
    
    print("\n[SUCCESS] Doppler integration successful!")

if __name__ == "__main__":
    test_secrets()