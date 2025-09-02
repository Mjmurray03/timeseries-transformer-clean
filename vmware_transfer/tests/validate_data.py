import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

def validate_stock_data():
    """Validate all stock data files and generate detailed report."""
    
    # Ensure we're in the right directory
    project_root = Path.cwd()
    if not (project_root / 'data').exists():
        print(f"Error: 'data' directory not found in {project_root}")
        print("   Make sure you're in the project root directory")
        return None
    
    results = {}
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA', 'TSLA', 'NFLX']
    
    print("=" * 60)
    print("STOCK DATA VALIDATION REPORT")
    print("=" * 60)
    
    for ticker in tickers:
        path = Path(f'data/raw/{ticker}.parquet')
        
        if not path.exists():
            results[ticker] = {"status": "MISSING", "path": str(path)}
            print(f"[MISSING] {ticker:6} | FILE NOT FOUND at {path}")
            continue
            
        try:
            df = pd.read_parquet(path)
            
            # Check for required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # Check for NaN values
            nan_counts = {}
            for col in required_cols:
                if col in df.columns:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        nan_counts[col] = int(nan_count)
            
            # Get date range
            if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                date_start = str(df.index.min())[:10]
                date_end = str(df.index.max())[:10]
            else:
                date_start = "Unknown"
                date_end = "Unknown"
            
            results[ticker] = {
                "status": "OK" if not missing_cols and not nan_counts else "WARNING",
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": date_start,
                    "end": date_end
                },
                "missing_columns": missing_cols,
                "nan_counts": nan_counts,
                "file_size_mb": round(path.stat().st_size / (1024*1024), 2)
            }
            
            status_label = "[OK]" if results[ticker]["status"] == "OK" else "[WARNING]"
            print(f"{status_label} {ticker:6} | {len(df):,} rows | {date_start} to {date_end} | {results[ticker]['file_size_mb']} MB")
            
            if missing_cols:
                print(f"   WARNING: Missing columns: {', '.join(missing_cols)}")
            if nan_counts:
                print(f"   WARNING: NaN values: {nan_counts}")
            
        except Exception as e:
            results[ticker] = {"status": "ERROR", "error": str(e)}
            print(f"[ERROR] {ticker:6} | ERROR: {str(e)[:50]}")
    
    # Save results
    output_dir = Path("test_results/data_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / f"data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("=" * 60)
    ok_count = sum(1 for r in results.values() if r.get("status") == "OK")
    warning_count = sum(1 for r in results.values() if r.get("status") == "WARNING")
    missing_count = sum(1 for r in results.values() if r.get("status") == "MISSING")
    error_count = sum(1 for r in results.values() if r.get("status") == "ERROR")
    
    print(f"\nSUMMARY:")
    print(f"   OK:       {ok_count}/{len(tickers)}")
    print(f"   Warnings: {warning_count}/{len(tickers)}")
    print(f"   Missing:  {missing_count}/{len(tickers)}")
    print(f"   Errors:   {error_count}/{len(tickers)}")
    
    print(f"\nReport saved to: {report_file}")
    
    if ok_count + warning_count == len(results):
        print("All required data files present and loadable!")
        return True
    else:
        print("Some data files have issues - check the report for details")
        return False

if __name__ == "__main__":
    result = validate_stock_data()
    sys.exit(0 if result else 1)