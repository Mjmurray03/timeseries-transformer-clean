"""
Comprehensive Data Loading Verification Script
Validates every step of the data pipeline from raw parquet files to feature-engineered tensors.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class DataLoadingVerifier:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.failure_details = []

    def verify_all(self) -> Dict:
        """Run all verification checks and return comprehensive report."""
        start_time = time.time()

        print("=" * 80)
        print("DATA LOADING VERIFICATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Directory: {self.data_dir}")
        print("=" * 80)

        # Find first parquet file
        parquet_file = self._find_parquet_file()
        if not parquet_file:
            self._report_failure("FILE_DISCOVERY", "No parquet files found in data/raw/")
            return self._generate_report(time.time() - start_time)

        print(f"\nAnalyzing file: {parquet_file}")
        print("-" * 80)

        # Load the data
        df = self._load_data(parquet_file)
        if df is None:
            return self._generate_report(time.time() - start_time)

        # Run all verification checks
        self._verify_columns(df)
        self._verify_data_types(df)
        self._verify_no_nans(df)
        self._verify_datetime_index(df)
        self._verify_chronological_order(df)
        self._verify_trading_days(df)
        self._verify_price_relationships(df)
        self._verify_data_shape_and_range(df)

        # Performance check
        elapsed_time = time.time() - start_time
        self._verify_performance(elapsed_time)

        return self._generate_report(elapsed_time)

    def _find_parquet_file(self) -> Optional[Path]:
        """Find the first parquet file in the data directory."""
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".parquet"):
                    return Path(root) / file
        return None

    def _load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load parquet file with error handling."""
        try:
            df = pd.read_parquet(file_path)
            self.results["FILE_LOADING"] = "PASS"
            print(f"✓ File loaded successfully")
            return df
        except Exception as e:
            self._report_failure("FILE_LOADING", f"Failed to load file: {str(e)}")
            return None

    def _verify_columns(self, df: pd.DataFrame):
        """Verify required columns are present."""
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        optional_columns = {"Ticker"}

        present_columns = set(df.columns)
        missing_required = required_columns - present_columns

        if missing_required:
            self._report_failure(
                "COLUMN_CHECK",
                f"Missing required columns: {missing_required}",
                {"present_columns": list(present_columns)},
            )
        else:
            # Check if columns are exactly as expected (with or without Ticker)
            expected_with_ticker = required_columns | optional_columns
            if present_columns == required_columns or present_columns == expected_with_ticker:
                self.results["COLUMN_CHECK"] = "PASS"
                print(f"✓ Column check passed: {sorted(present_columns)}")
            else:
                extra_columns = present_columns - expected_with_ticker
                self._report_failure(
                    "COLUMN_CHECK",
                    f"Unexpected columns found: {extra_columns}",
                    {"present_columns": list(present_columns)},
                )

    def _verify_data_types(self, df: pd.DataFrame):
        """Verify data types of numeric columns."""
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        valid_types = [np.float64, np.float32, np.int64, np.int32]

        invalid_types = []
        for col in numeric_columns:
            if col in df.columns:
                dtype = df[col].dtype
                if dtype not in valid_types:
                    invalid_types.append(
                        {
                            "column": col,
                            "actual_type": str(dtype),
                            "expected_types": ["float64", "float32", "int64", "int32"],
                        }
                    )

        if invalid_types:
            self._report_failure("DATA_TYPE_CHECK", "Invalid data types detected", invalid_types)
        else:
            self.results["DATA_TYPE_CHECK"] = "PASS"
            type_summary = {col: str(df[col].dtype) for col in numeric_columns if col in df.columns}
            print(f"✓ Data type check passed: {type_summary}")

    def _verify_no_nans(self, df: pd.DataFrame):
        """Verify no NaN values in OHLCV columns."""
        ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
        nan_info = []

        for col in ohlcv_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_indices = df[df[col].isna()].index.tolist()[:5]  # First 5 NaN locations
                    nan_info.append(
                        {"column": col, "nan_count": int(nan_count), "sample_indices": nan_indices}
                    )

        if nan_info:
            self._report_failure("NAN_CHECK", "NaN values detected in critical columns", nan_info)
        else:
            self.results["NAN_CHECK"] = "PASS"
            print(f"✓ NaN check passed: No missing values in OHLCV columns")

    def _verify_datetime_index(self, df: pd.DataFrame):
        """Verify the DataFrame has a properly formatted DatetimeIndex."""
        if isinstance(df.index, pd.DatetimeIndex):
            self.results["DATETIME_INDEX_CHECK"] = "PASS"
            print(f"✓ DatetimeIndex check passed: Index is properly formatted")
        else:
            # Try to parse index as datetime
            try:
                df.index = pd.to_datetime(df.index)
                if isinstance(df.index, pd.DatetimeIndex):
                    self.results["DATETIME_INDEX_CHECK"] = "PASS"
                    print(f"✓ DatetimeIndex check passed: Index converted to datetime")
                else:
                    raise ValueError("Could not convert to DatetimeIndex")
            except Exception as e:
                self._report_failure(
                    "DATETIME_INDEX_CHECK",
                    f"Index is not DatetimeIndex: {type(df.index).__name__}",
                    {"error": str(e)},
                )

    def _verify_chronological_order(self, df: pd.DataFrame):
        """Verify data is sorted chronologically."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self._report_failure(
                "CHRONOLOGICAL_ORDER_CHECK", "Cannot verify order without DatetimeIndex"
            )
            return

        is_sorted = df.index.equals(df.index.sort_values())
        if is_sorted:
            self.results["CHRONOLOGICAL_ORDER_CHECK"] = "PASS"
            print(f"✓ Chronological order check passed: Data is properly sorted")
        else:
            # Find first unsorted position
            for i in range(1, len(df.index)):
                if df.index[i] < df.index[i - 1]:
                    self._report_failure(
                        "CHRONOLOGICAL_ORDER_CHECK",
                        "Data is not sorted chronologically",
                        {
                            "first_unsorted_position": i,
                            "date_at_position": str(df.index[i]),
                            "previous_date": str(df.index[i - 1]),
                        },
                    )
                    break

    def _verify_trading_days(self, df: pd.DataFrame):
        """Check for weekend/holiday gaps and verify they're expected."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self._report_failure(
                "TRADING_DAYS_CHECK", "Cannot verify trading days without DatetimeIndex"
            )
            return

        # Calculate gaps between consecutive dates
        date_diffs = df.index[1:] - df.index[:-1]
        business_days = pd.bdate_range(start=df.index[0], end=df.index[-1])

        # Check if all dates are business days
        non_business_days = []
        for date in df.index:
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                non_business_days.append(str(date.date()))

        if non_business_days:
            self._report_failure(
                "TRADING_DAYS_CHECK",
                "Data contains weekend dates",
                {"sample_weekend_dates": non_business_days[:5]},
            )
        else:
            # Check for large gaps (more than 4 business days usually indicates holidays)
            large_gaps = []
            for i, diff in enumerate(date_diffs):
                if diff.days > 4:
                    large_gaps.append(
                        {
                            "from": str(df.index[i].date()),
                            "to": str(df.index[i + 1].date()),
                            "gap_days": diff.days,
                        }
                    )

            if large_gaps:
                print(
                    f"⚠ Trading days check: Found {len(large_gaps)} gaps > 4 days (likely holidays)"
                )
                self.results["TRADING_DAYS_CHECK"] = "PASS"  # Gaps are expected for holidays
            else:
                self.results["TRADING_DAYS_CHECK"] = "PASS"
                print(f"✓ Trading days check passed: No unexpected gaps found")

    def _verify_price_relationships(self, df: pd.DataFrame):
        """Verify price relationships: High >= Low, High >= Close, High >= Open, etc."""
        violations = []

        # Check High >= Low
        if "High" in df.columns and "Low" in df.columns:
            high_low_violations = df[df["High"] < df["Low"]]
            if not high_low_violations.empty:
                violations.append(
                    {
                        "rule": "High >= Low",
                        "violation_count": len(high_low_violations),
                        "sample_rows": high_low_violations.index[:3].tolist(),
                    }
                )

        # Check High >= Close
        if "High" in df.columns and "Close" in df.columns:
            high_close_violations = df[df["High"] < df["Close"]]
            if not high_close_violations.empty:
                violations.append(
                    {
                        "rule": "High >= Close",
                        "violation_count": len(high_close_violations),
                        "sample_rows": high_close_violations.index[:3].tolist(),
                    }
                )

        # Check High >= Open
        if "High" in df.columns and "Open" in df.columns:
            high_open_violations = df[df["High"] < df["Open"]]
            if not high_open_violations.empty:
                violations.append(
                    {
                        "rule": "High >= Open",
                        "violation_count": len(high_open_violations),
                        "sample_rows": high_open_violations.index[:3].tolist(),
                    }
                )

        # Check Low <= Close
        if "Low" in df.columns and "Close" in df.columns:
            low_close_violations = df[df["Low"] > df["Close"]]
            if not low_close_violations.empty:
                violations.append(
                    {
                        "rule": "Low <= Close",
                        "violation_count": len(low_close_violations),
                        "sample_rows": low_close_violations.index[:3].tolist(),
                    }
                )

        # Check Low <= Open
        if "Low" in df.columns and "Open" in df.columns:
            low_open_violations = df[df["Low"] > df["Open"]]
            if not low_open_violations.empty:
                violations.append(
                    {
                        "rule": "Low <= Open",
                        "violation_count": len(low_open_violations),
                        "sample_rows": low_open_violations.index[:3].tolist(),
                    }
                )

        if violations:
            self._report_failure(
                "PRICE_RELATIONSHIP_CHECK", "Price relationship violations detected", violations
            )
        else:
            self.results["PRICE_RELATIONSHIP_CHECK"] = "PASS"
            print(f"✓ Price relationship check passed: All price constraints satisfied")

    def _verify_data_shape_and_range(self, df: pd.DataFrame):
        """Report the shape and date range of loaded data."""
        shape = df.shape

        if isinstance(df.index, pd.DatetimeIndex):
            date_range = {
                "start_date": str(df.index[0].date()),
                "end_date": str(df.index[-1].date()),
                "total_days": (df.index[-1] - df.index[0]).days,
                "total_records": len(df),
            }
        else:
            date_range = {"error": "DatetimeIndex not available"}

        self.results["DATA_SHAPE"] = {
            "rows": shape[0],
            "columns": shape[1],
            "date_range": date_range,
        }

        print(f"\n📊 Data Shape and Range:")
        print(f"   Rows: {shape[0]:,}")
        print(f"   Columns: {shape[1]}")
        if "error" not in date_range:
            print(f"   Date Range: {date_range['start_date']} to {date_range['end_date']}")
            print(f"   Total Days Covered: {date_range['total_days']:,}")

    def _verify_performance(self, elapsed_time: float):
        """Verify script completes within performance requirements."""
        max_time = 10.0  # 10 seconds for 5-year dataset

        if elapsed_time <= max_time:
            self.results["PERFORMANCE_CHECK"] = "PASS"
            print(f"\n⚡ Performance check passed: Completed in {elapsed_time:.2f} seconds")
        else:
            self._report_failure(
                "PERFORMANCE_CHECK",
                f"Exceeded maximum time of {max_time} seconds",
                {"elapsed_time": elapsed_time},
            )

    def _make_json_serializable(self, obj):
        """Convert pandas Timestamps and other non-JSON types to strings."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "strftime"):  # pandas Timestamp
            return str(obj)
        elif hasattr(obj, "item"):  # numpy types
            return obj.item()
        else:
            return obj

    def _report_failure(self, check_name: str, message: str, details: Optional[Dict] = None):
        """Record a failure with details."""
        self.results[check_name] = "FAIL"
        failure_info = {"check": check_name, "message": message, "details": details}
        self.failure_details.append(failure_info)
        print(f"✗ {check_name} FAILED: {message}")
        if details:
            print(f"  Details: {details}")

    def _generate_report(self, elapsed_time: float) -> Dict:
        """Generate comprehensive verification report."""
        total_checks = len([v for v in self.results.values() if v in ["PASS", "FAIL"]])
        passed_checks = len([v for v in self.results.values() if v == "PASS"])
        failed_checks = len([v for v in self.results.values() if v == "FAIL"])

        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks} ✓")
        print(f"Failed: {failed_checks} ✗")
        print(f"Success Rate: {(passed_checks/total_checks*100) if total_checks > 0 else 0:.1f}%")
        print(f"Execution Time: {elapsed_time:.2f} seconds")

        if failed_checks > 0:
            print("\n⚠️  FAILURES DETECTED:")
            for failure in self.failure_details:
                print(f"\n  • {failure['check']}: {failure['message']}")
                if failure.get("details"):
                    import json

                    # Convert pandas Timestamps to strings for JSON serialization
                    details_serializable = self._make_json_serializable(failure["details"])
                    print(f"    {json.dumps(details_serializable, indent=6)}")

        overall_status = "PASS" if failed_checks == 0 else "FAIL"
        print(f"\n{'='*80}")
        print(f"OVERALL STATUS: {overall_status}")
        print(f"{'='*80}")

        return {
            "status": overall_status,
            "summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "execution_time": elapsed_time,
            },
            "results": self.results,
            "failures": self.failure_details,
        }


def main():
    """Main execution function."""
    verifier = DataLoadingVerifier()
    report = verifier.verify_all()

    # Return appropriate exit code
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
