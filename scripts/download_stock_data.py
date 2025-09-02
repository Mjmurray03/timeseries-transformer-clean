import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data():
    """Download historical stock data for all required tickers."""

    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # List of tickers to download
    tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "NFLX"]

    # Date range (5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    print("=" * 60)
    print("DOWNLOADING STOCK DATA")
    print("=" * 60)
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Output Directory: {data_dir.absolute()}")
    print("-" * 60)

    successful = []
    failed = []

    for ticker in tickers:
        try:
            print(f"\nDownloading {ticker}...", end="")

            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f" [ERROR] No data retrieved")
                failed.append(ticker)
                continue

            # Clean column names (remove spaces)
            df.columns = df.columns.str.replace(" ", "")

            # Ensure we have required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                print(f" [WARNING] Missing required columns")
                failed.append(ticker)
                continue

            # Save as parquet
            output_path = data_dir / f"{ticker}.parquet"
            df.to_parquet(output_path)

            # Verify file was created
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            print(f" [OK] Done! ({len(df):,} rows, {file_size_mb:.2f} MB)")
            successful.append(ticker)

        except Exception as e:
            print(f" [ERROR] Error: {str(e)}")
            failed.append(ticker)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"[OK] Successful: {len(successful)}/{len(tickers)} - {', '.join(successful)}")
    if failed:
        print(f"[ERROR] Failed: {len(failed)}/{len(tickers)} - {', '.join(failed)}")

    return len(successful) == len(tickers)


if __name__ == "__main__":
    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("[ERROR] yfinance not installed. Installing now...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        print("[OK] yfinance installed. Please run the script again.")
        sys.exit(1)

    success = download_stock_data()
    sys.exit(0 if success else 1)
