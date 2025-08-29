#!/usr/bin/env python3
"""
Data Download Script

Simple script to download stock data for testing and training.
"""

import yfinance as yf
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def download_stock_data(ticker: str, period: str = "5y", save_path: str = "data/sample") -> None:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1y, 2y, 5y, max)
        save_path: Directory to save the data
    """
    print(f"Downloading {ticker} data for period: {period}")
    
    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        filename = f"{ticker}_{period}.parquet"
        filepath = Path(save_path) / filename
        df.to_parquet(filepath, index=False)
        
        print(f"[OK] Saved {len(df)} rows to: {filepath}")
        print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
    except Exception as e:
        print(f"[ERROR] Error downloading {ticker}: {e}")


def create_sample_data(ticker: str = "AAPL", num_rows: int = 100, save_path: str = "data/sample") -> None:
    """
    Create a small sample dataset for quick testing.
    
    Args:
        ticker: Stock ticker symbol  
        num_rows: Number of rows to include
        save_path: Directory to save the sample
    """
    print(f"Creating sample dataset for {ticker} with {num_rows} rows")
    
    try:
        # Download recent data
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            print(f"No data found for {ticker}")
            return
        
        # Take the last num_rows
        df_sample = df.tail(num_rows).copy()
        df_sample = df_sample.reset_index()
        
        # Ensure directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        filename = f"{ticker}_sample.parquet"
        filepath = Path(save_path) / filename
        df_sample.to_parquet(filepath, index=False)
        
        print(f"[OK] Created sample dataset: {filepath}")
        print(f"   Rows: {len(df_sample)}")
        print(f"   Date range: {df_sample['Date'].min().date()} to {df_sample['Date'].max().date()}")
        print(f"   Columns: {list(df_sample.columns)}")
        
    except Exception as e:
        print(f"[ERROR] Error creating sample for {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Download stock data')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='5y', help='Time period (1y, 2y, 5y, max)')
    parser.add_argument('--sample-only', action='store_true', help='Only create sample data')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of rows in sample')
    parser.add_argument('--output-dir', type=str, default='data/sample', help='Output directory')
    args = parser.parse_args()
    
    print(f"{'='*50}")
    print(f"STOCK DATA DOWNLOADER")
    print(f"{'='*50}")
    
    if args.sample_only:
        # Create sample data only
        create_sample_data(args.ticker, args.sample_size, args.output_dir)
    else:
        # Download full dataset
        download_stock_data(args.ticker, args.period, args.output_dir)
        
        # Also create sample
        create_sample_data(args.ticker, args.sample_size, args.output_dir)
    
    print(f"\n[OK] Download complete!")


if __name__ == "__main__":
    main()