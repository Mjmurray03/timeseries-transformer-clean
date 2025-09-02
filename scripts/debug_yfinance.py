#!/usr/bin/env python3
"""
Debug script to see what yfinance actually returns.
"""

from datetime import date

import yfinance as yf

# Download some data to see the structure
print("Downloading AAPL data...")
data = yf.download("AAPL", start="2024-01-01", end="2024-01-31", progress=False)

print(f"Type: {type(data)}")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns}")
print(f"Index: {data.index}")
print(f"Empty: {data.empty}")

print("\nFirst few rows:")
print(data.head())

print("\nColumn types:")
print(data.dtypes)

print("\nChecking for MultiIndex:")
print(f"Columns is MultiIndex: {hasattr(data.columns, 'levels')}")
if hasattr(data.columns, "levels"):
    print(f"Column levels: {data.columns.levels}")

print("\nTrying to access 'Close' column:")
try:
    close_data = data["Close"]
    print(f"Close data type: {type(close_data)}")
    print(f"Close data shape: {close_data.shape}")
    print(f"Close data empty: {close_data.empty}")
except Exception as e:
    print(f"Error accessing Close: {e}")

print("\nTrying boolean operations:")
try:
    if "Close" in data.columns:
        result = data["Close"] > 100
        print(f"Boolean operation result type: {type(result)}")
        print(f"Any values > 100: {result.any()}")
except Exception as e:
    print(f"Error in boolean operation: {e}")
