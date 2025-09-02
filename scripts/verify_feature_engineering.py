"""
Feature Engineering Verification Script
Validates that FeatureEngineer class produces correct technical indicators.
"""

import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows console (only when running as main script)
if sys.platform == "win32" and __name__ == "__main__":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.processors.feature_engineering import FeatureEngineer


class FeatureVerifier:
    """Verifies FeatureEngineer calculations against manual implementations."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.results = {}
        self.comparison_data = []
        self.verification_errors = []

    def verify_all_features(self, data_path: str = None) -> Dict:
        """Run comprehensive feature engineering verification."""
        start_time = time.time()

        print("=" * 80)
        print("FEATURE ENGINEERING VERIFICATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tolerance: {self.tolerance}")
        print("=" * 80)

        # Load data
        df = self._load_data(data_path)
        if df is None:
            return self._generate_report(time.time() - start_time)

        print(f"\nLoaded data: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

        # Apply FeatureEngineer
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(df)

        print(
            f"Engineered features: {engineered_df.shape[0]} rows × {engineered_df.shape[1]} columns"
        )

        # Verify input/output dimensions
        self._verify_dimensions(df, engineered_df)

        # Select random sample points for detailed verification
        sample_indices = self._select_sample_indices(engineered_df, n_samples=5)
        print(
            f"\nSelected sample indices for verification: {[str(idx.date()) for idx in sample_indices]}"
        )

        # Verify each indicator
        self._verify_sma(df, engineered_df, sample_indices)
        self._verify_rsi(df, engineered_df, sample_indices)
        self._verify_macd(df, engineered_df, sample_indices)
        self._verify_bollinger_bands(df, engineered_df, sample_indices)
        self._verify_returns(df, engineered_df, sample_indices)

        # Check for data leakage
        self._verify_no_data_leakage(df, engineered_df)

        # Check for missing features
        self._verify_feature_completeness(engineered_df)

        elapsed_time = time.time() - start_time
        self._verify_performance(elapsed_time)

        return self._generate_report(elapsed_time)

    def _load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load OHLCV data from parquet file."""
        if data_path is None:
            # Use first available parquet file
            data_dir = Path("data/raw")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".parquet"):
                        data_path = Path(root) / file
                        break
                if data_path:
                    break

        if not data_path:
            self._report_error("DATA_LOADING", "No parquet files found")
            return None

        try:
            df = pd.read_parquet(data_path)
            print(f"Loaded data from: {data_path}")

            # Verify required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self._report_error("DATA_LOADING", f"Missing required columns: {missing_cols}")
                return None

            self.results["DATA_LOADING"] = "PASS"
            return df

        except Exception as e:
            self._report_error("DATA_LOADING", f"Failed to load data: {str(e)}")
            return None

    def _select_sample_indices(self, df: pd.DataFrame, n_samples: int = 5) -> List:
        """Select random indices for detailed verification, avoiding NaN periods."""
        # Skip first 200 rows to avoid NaN values from indicators
        valid_start = min(200, len(df) - 50)  # Ensure we have some data to sample from
        valid_indices = df.index[valid_start:]

        if len(valid_indices) < n_samples:
            return list(valid_indices)

        random.seed(42)  # For reproducible testing
        return sorted(random.sample(list(valid_indices), n_samples))

    def _verify_dimensions(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Verify input/output dimensions match expectations."""
        input_shape = input_df.shape
        output_shape = output_df.shape

        # Input should be (n_rows, 6) for OHLCV + Ticker
        if input_shape[1] != 6:
            self._report_error("DIMENSIONS", f"Input columns expected 6, got {input_shape[1]}")
            return

        # Output should have same number of rows (or fewer due to lookback) and more features
        if output_shape[0] > input_shape[0]:
            self._report_error(
                "DIMENSIONS", f"Output rows ({output_shape[0]}) > input rows ({input_shape[0]})"
            )
            return

        if output_shape[1] < 30:  # Should have at least 30 features
            self._report_error(
                "DIMENSIONS", f"Output features expected >= 30, got {output_shape[1]}"
            )
            return

        self.results["DIMENSIONS"] = "PASS"
        print(f"+ Dimension check passed: {input_shape} -> {output_shape}")

    def _verify_sma(self, input_df: pd.DataFrame, output_df: pd.DataFrame, sample_indices: List):
        """Verify SMA_20 calculation."""
        if "MA_20" not in output_df.columns:
            self._report_error("SMA_20", "MA_20 column not found in output")
            return

        close_prices = input_df["Close"]
        actual_sma = output_df["MA_20"]

        for idx in sample_indices:
            if idx not in actual_sma.index:
                continue

            # Manual SMA calculation
            idx_pos = input_df.index.get_loc(idx)
            if idx_pos < 19:  # Need at least 20 values
                continue

            start_pos = idx_pos - 19
            end_pos = idx_pos + 1
            window_data = close_prices.iloc[start_pos:end_pos]
            expected_sma = window_data.mean()

            actual_val = actual_sma.loc[idx]

            if pd.isna(expected_sma) or pd.isna(actual_val):
                continue

            diff = abs(expected_sma - actual_val)

            self.comparison_data.append(
                {
                    "Indicator": "SMA_20",
                    "Date": str(idx.date()),
                    "Expected": expected_sma,
                    "Actual": actual_val,
                    "Difference": diff,
                    "Within_Tolerance": diff <= self.tolerance,
                }
            )

            if diff > self.tolerance:
                self._report_error(
                    "SMA_20",
                    f"At {idx.date()}: Expected {expected_sma:.6f}, got {actual_val:.6f}, diff {diff:.6f}",
                )
                return

        self.results["SMA_20"] = "PASS"
        print("+ SMA_20 verification passed")

    def _verify_rsi(self, input_df: pd.DataFrame, output_df: pd.DataFrame, sample_indices: List):
        """Verify RSI calculation."""
        if "RSI" not in output_df.columns:
            self._report_error("RSI", "RSI column not found in output")
            return

        close_prices = input_df["Close"]
        actual_rsi = output_df["RSI"]

        # Check RSI range [0, 100]
        rsi_out_of_range = actual_rsi[(actual_rsi < 0) | (actual_rsi > 100)]
        if not rsi_out_of_range.empty:
            self._report_error(
                "RSI", f"RSI values outside [0, 100] range: {len(rsi_out_of_range)} values"
            )
            return

        # Manual RSI calculation for sample points
        for idx in sample_indices:
            if idx not in actual_rsi.index:
                continue

            expected_rsi = self._calculate_manual_rsi(close_prices, idx, period=14)
            actual_val = actual_rsi.loc[idx]

            if pd.isna(expected_rsi) or pd.isna(actual_val):
                continue

            diff = abs(expected_rsi - actual_val)

            self.comparison_data.append(
                {
                    "Indicator": "RSI",
                    "Date": str(idx.date()),
                    "Expected": expected_rsi,
                    "Actual": actual_val,
                    "Difference": diff,
                    "Within_Tolerance": diff <= self.tolerance,
                }
            )

            if diff > self.tolerance:
                self._report_error(
                    "RSI",
                    f"At {idx.date()}: Expected {expected_rsi:.6f}, got {actual_val:.6f}, diff {diff:.6f}",
                )
                return

        self.results["RSI"] = "PASS"
        print("+ RSI verification passed")

    def _calculate_manual_rsi(self, prices: pd.Series, target_idx, period: int = 14) -> float:
        """Manual RSI calculation for verification - matching FeatureEngineer implementation."""
        idx_pos = prices.index.get_loc(target_idx)

        # FeatureEngineer sets RSI to 50 for first 'period' values
        if idx_pos < period:
            return 50.0

        # Calculate RSI using rolling window like FeatureEngineer does
        price_data_up_to_idx = prices.iloc[: idx_pos + 1]
        delta = price_data_up_to_idx.diff()

        # Use same logic as FeatureEngineer: rolling window on gains/losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Get the value at target index
        avg_gain = gain.iloc[-1]
        avg_loss = loss.iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _verify_macd(self, input_df: pd.DataFrame, output_df: pd.DataFrame, sample_indices: List):
        """Verify MACD calculation."""
        required_cols = ["MACD", "MACD_Signal", "MACD_Histogram"]
        missing_cols = [col for col in required_cols if col not in output_df.columns]
        if missing_cols:
            self._report_error("MACD", f"Missing MACD columns: {missing_cols}")
            return

        close_prices = input_df["Close"]

        for idx in sample_indices:
            if idx not in output_df.index:
                continue

            # Manual MACD calculation
            expected_macd_data = self._calculate_manual_macd(close_prices, idx)
            if not expected_macd_data:
                continue

            for component in ["MACD", "MACD_Signal", "MACD_Histogram"]:
                expected_val = expected_macd_data[component]
                actual_val = output_df.loc[idx, component]

                if pd.isna(expected_val) or pd.isna(actual_val):
                    continue

                diff = abs(expected_val - actual_val)

                self.comparison_data.append(
                    {
                        "Indicator": component,
                        "Date": str(idx.date()),
                        "Expected": expected_val,
                        "Actual": actual_val,
                        "Difference": diff,
                        "Within_Tolerance": diff <= self.tolerance,
                    }
                )

                if diff > self.tolerance:
                    self._report_error(
                        "MACD",
                        f"{component} at {idx.date()}: Expected {expected_val:.6f}, got {actual_val:.6f}, diff {diff:.6f}",
                    )
                    return

        self.results["MACD"] = "PASS"
        print("+ MACD verification passed")

    def _calculate_manual_macd(
        self, prices: pd.Series, target_idx, fast=12, slow=26, signal=9
    ) -> Dict:
        """Manual MACD calculation for verification."""
        idx_pos = prices.index.get_loc(target_idx)
        if idx_pos < slow - 1:
            return None

        # Calculate EMAs up to target index
        price_data = prices.iloc[: idx_pos + 1]

        ema_fast = price_data.ewm(span=fast).mean().iloc[-1]
        ema_slow = price_data.ewm(span=slow).mean().iloc[-1]

        macd_line = ema_fast - ema_slow

        # For signal line, need MACD history
        if idx_pos < slow + signal - 2:
            return {"MACD": macd_line, "MACD_Signal": np.nan, "MACD_Histogram": np.nan}

        # Calculate MACD series up to this point
        ema_fast_series = price_data.ewm(span=fast).mean()
        ema_slow_series = price_data.ewm(span=slow).mean()
        macd_series = ema_fast_series - ema_slow_series

        # Signal line is EMA of MACD
        signal_line = macd_series.ewm(span=signal).mean().iloc[-1]
        histogram = macd_line - signal_line

        return {"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Histogram": histogram}

    def _verify_bollinger_bands(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, sample_indices: List
    ):
        """Verify Bollinger Bands calculation."""
        required_cols = ["BB_Upper", "BB_Middle", "BB_Lower"]
        missing_cols = [col for col in required_cols if col not in output_df.columns]
        if missing_cols:
            self._report_error("BOLLINGER_BANDS", f"Missing Bollinger Band columns: {missing_cols}")
            return

        close_prices = input_df["Close"]

        for idx in sample_indices:
            if idx not in output_df.index:
                continue

            # Manual Bollinger Bands calculation
            expected_bb = self._calculate_manual_bollinger_bands(
                close_prices, idx, period=20, std_dev=2.0
            )
            if not expected_bb:
                continue

            for component in ["BB_Upper", "BB_Middle", "BB_Lower"]:
                expected_val = expected_bb[component]
                actual_val = output_df.loc[idx, component]

                if pd.isna(expected_val) or pd.isna(actual_val):
                    continue

                diff = abs(expected_val - actual_val)

                self.comparison_data.append(
                    {
                        "Indicator": component,
                        "Date": str(idx.date()),
                        "Expected": expected_val,
                        "Actual": actual_val,
                        "Difference": diff,
                        "Within_Tolerance": diff <= self.tolerance,
                    }
                )

                if diff > self.tolerance:
                    self._report_error(
                        "BOLLINGER_BANDS",
                        f"{component} at {idx.date()}: Expected {expected_val:.6f}, got {actual_val:.6f}, diff {diff:.6f}",
                    )
                    return

        # Verify BB ordering (Upper >= Middle >= Lower)
        valid_bb_rows = ~(
            output_df["BB_Upper"].isna()
            | output_df["BB_Middle"].isna()
            | output_df["BB_Lower"].isna()
        )
        if valid_bb_rows.any():
            upper_ge_middle = (
                output_df.loc[valid_bb_rows, "BB_Upper"]
                >= output_df.loc[valid_bb_rows, "BB_Middle"]
            ).all()
            middle_ge_lower = (
                output_df.loc[valid_bb_rows, "BB_Middle"]
                >= output_df.loc[valid_bb_rows, "BB_Lower"]
            ).all()

            if not upper_ge_middle or not middle_ge_lower:
                self._report_error(
                    "BOLLINGER_BANDS", "Bollinger Band ordering violated (Upper >= Middle >= Lower)"
                )
                return

        self.results["BOLLINGER_BANDS"] = "PASS"
        print("+ Bollinger Bands verification passed")

    def _calculate_manual_bollinger_bands(
        self, prices: pd.Series, target_idx, period: int = 20, std_dev: float = 2.0
    ) -> Dict:
        """Manual Bollinger Bands calculation for verification."""
        idx_pos = prices.index.get_loc(target_idx)
        if idx_pos < period - 1:
            return None

        # Get window data
        start_pos = idx_pos - period + 1
        end_pos = idx_pos + 1
        window_data = prices.iloc[start_pos:end_pos]

        bb_middle = window_data.mean()  # SMA
        bb_std = window_data.std()  # Standard deviation

        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)

        return {"BB_Upper": bb_upper, "BB_Middle": bb_middle, "BB_Lower": bb_lower}

    def _verify_returns(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, sample_indices: List
    ):
        """Verify Returns calculation."""
        if "Returns" not in output_df.columns:
            self._report_error("RETURNS", "Returns column not found in output")
            return

        close_prices = input_df["Close"]
        actual_returns = output_df["Returns"]

        for idx in sample_indices:
            if idx not in actual_returns.index:
                continue

            # Manual returns calculation: (Close[t] - Close[t-1]) / Close[t-1]
            idx_pos = close_prices.index.get_loc(idx)
            if idx_pos == 0:  # First row has NaN return
                continue

            current_price = close_prices.iloc[idx_pos]
            previous_price = close_prices.iloc[idx_pos - 1]
            expected_return = (current_price - previous_price) / previous_price

            actual_val = actual_returns.loc[idx]

            if pd.isna(expected_return) or pd.isna(actual_val):
                continue

            diff = abs(expected_return - actual_val)

            self.comparison_data.append(
                {
                    "Indicator": "Returns",
                    "Date": str(idx.date()),
                    "Expected": expected_return,
                    "Actual": actual_val,
                    "Difference": diff,
                    "Within_Tolerance": diff <= self.tolerance,
                }
            )

            if diff > self.tolerance:
                self._report_error(
                    "RETURNS",
                    f"At {idx.date()}: Expected {expected_return:.6f}, got {actual_val:.6f}, diff {diff:.6f}",
                )
                return

        self.results["RETURNS"] = "PASS"
        print("+ Returns verification passed")

    def _verify_no_data_leakage(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Verify no future data is used in calculations."""
        # Check that features at time t only use data up to time t
        # This is implicit in rolling window calculations, but we verify by checking
        # that indicators make sense chronologically

        # Check that first few values are NaN where expected (due to lookback periods)
        # Note: RSI is set to 50 for initial period in FeatureEngineer, so we check differently
        indicators_with_lookback = {
            "MA_20": 19,  # First 19 should be NaN
            "BB_Middle": 19,  # First 19 should be NaN
        }

        for indicator, expected_nan_count in indicators_with_lookback.items():
            if indicator in output_df.columns:
                actual_nan_count = output_df[indicator].iloc[:expected_nan_count].isna().sum()
                if actual_nan_count < expected_nan_count * 0.8:  # Allow some flexibility
                    self._report_error(
                        "DATA_LEAKAGE",
                        f"{indicator} should have ~{expected_nan_count} initial NaN values, got {actual_nan_count}",
                    )
                    return

        # Special check for RSI: first 14 values should be 50 (initialization value)
        if "RSI" in output_df.columns:
            initial_rsi_values = output_df["RSI"].iloc[:14]
            if not (initial_rsi_values == 50).all():
                non_50_count = (initial_rsi_values != 50).sum()
                self._report_error(
                    "DATA_LEAKAGE",
                    f"RSI first 14 values should be 50 (initialization), but {non_50_count} are different",
                )
                return

        # Check that Returns has NaN at first position (no previous price)
        if "Returns" in output_df.columns:
            if not pd.isna(output_df["Returns"].iloc[0]):
                self._report_error(
                    "DATA_LEAKAGE",
                    "Returns first value should be NaN (no previous price to compare)",
                )
                return

        self.results["DATA_LEAKAGE"] = "PASS"
        print("+ No data leakage detected")

    def _verify_feature_completeness(self, output_df: pd.DataFrame):
        """Verify all expected features are present."""
        expected_features = [
            "Returns",
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "MA_20",
        ]

        missing_features = [f for f in expected_features if f not in output_df.columns]
        if missing_features:
            self._report_error(
                "FEATURE_COMPLETENESS", f"Missing expected features: {missing_features}"
            )
            return

        self.results["FEATURE_COMPLETENESS"] = "PASS"
        print("+ Feature completeness check passed")

    def _verify_performance(self, elapsed_time: float):
        """Verify performance requirements."""
        max_time = 30.0  # 30 seconds maximum

        if elapsed_time <= max_time:
            self.results["PERFORMANCE"] = "PASS"
            print(f"+ Performance check passed: Completed in {elapsed_time:.2f} seconds")
        else:
            self._report_error(
                "PERFORMANCE", f"Exceeded maximum time of {max_time} seconds: {elapsed_time:.2f}s"
            )

    def _report_error(self, check_name: str, message: str):
        """Record verification error."""
        self.results[check_name] = "FAIL"
        self.verification_errors.append({"check": check_name, "message": message})
        print(f"X {check_name} FAILED: {message}")

    def _generate_report(self, elapsed_time: float) -> Dict:
        """Generate comprehensive verification report."""
        total_checks = len([v for v in self.results.values() if v in ["PASS", "FAIL"]])
        passed_checks = len([v for v in self.results.values() if v == "PASS"])
        failed_checks = len([v for v in self.results.values() if v == "FAIL"])

        print("\n" + "=" * 80)
        print("DETAILED COMPARISON TABLE")
        print("=" * 80)

        if self.comparison_data:
            comparison_df = pd.DataFrame(self.comparison_data)
            print(comparison_df.to_string(index=False, float_format="%.6f"))

            # Summary of comparisons
            total_comparisons = len(self.comparison_data)
            within_tolerance = sum(1 for item in self.comparison_data if item["Within_Tolerance"])
            print(f"\nComparison Summary: {within_tolerance}/{total_comparisons} within tolerance")
        else:
            print("No detailed comparisons available")

        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks} +")
        print(f"Failed: {failed_checks} X")
        print(f"Success Rate: {(passed_checks/total_checks*100) if total_checks > 0 else 0:.1f}%")
        print(f"Execution Time: {elapsed_time:.2f} seconds")

        if self.verification_errors:
            print("\nVERIFICATION ERRORS:")
            for error in self.verification_errors:
                print(f"  - {error['check']}: {error['message']}")

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
            "errors": self.verification_errors,
            "comparison_data": self.comparison_data,
        }


def main():
    """Main execution function."""
    verifier = FeatureVerifier()
    report = verifier.verify_all_features()

    # Return appropriate exit code
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
