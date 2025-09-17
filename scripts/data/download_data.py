import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_stock_data(tickers=None):
    """Download historical stock data for all required tickers."""

    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # List of tickers to download
    if tickers is None:
        tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "TSLA", "NFLX"]
    elif isinstance(tickers, str):
        tickers = [tickers]

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

            # Save as both parquet and CSV for compatibility
            parquet_path = data_dir / f"{ticker}.parquet"
            csv_path = data_dir / f"{ticker}.csv"

            df.to_parquet(parquet_path)
            df.to_csv(csv_path)

            output_path = parquet_path

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


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Download historical stock data for backtesting and training"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Specific ticker to download (e.g., AAPL). If not specified, downloads all default tickers."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Multiple tickers to download (e.g., AAPL MSFT NVDA)"
    )

    args = parser.parse_args()

    # Determine which tickers to download
    tickers_to_download = None
    if args.ticker:
        tickers_to_download = [args.ticker.upper()]
    elif args.tickers:
        tickers_to_download = [t.upper() for t in args.tickers]

    success = download_stock_data(tickers_to_download)
    return 0 if success else 1


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

    sys.exit(main())
