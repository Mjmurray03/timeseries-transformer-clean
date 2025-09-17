#!/usr/bin/env python3
"""
TimeSeries Transformer - Complete Demo Pipeline

Complete demo script that runs the entire pipeline from data download to predictions.
No configuration needed - works out of the box with intelligent defaults.

Features:
- Automatic environment detection (GPU/CPU, dependencies)
- Robust error handling with helpful suggestions
- Progress tracking and status updates
- Comprehensive output with next steps guidance

Usage:
    python scripts/run_demo.py                    # Run full demo with AAPL
    python scripts/run_demo.py --ticker MSFT      # Run demo with different ticker
    python scripts/run_demo.py --quick            # Quick demo (5 epochs)
    python scripts/run_demo.py --full             # Full demo (50 epochs)

Outputs:
    - data/raw/{ticker}.csv - Downloaded stock data
    - models/model_{ticker}_best.pt - Trained model
    - scalers/scaler_{ticker}.json - Scaler parameters
    - predictions/predictions_{timestamp}.json - Generated predictions
    - logs/demo_{timestamp}.log - Complete execution log
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class DemoRunner:
    """Production-ready demo runner with comprehensive error handling and monitoring."""

    def __init__(self, ticker: str = "AAPL", quick_mode: bool = False, full_mode: bool = False):
        self.ticker = ticker.upper()
        self.quick_mode = quick_mode
        self.full_mode = full_mode
        self.start_time = datetime.now()
        self.steps_completed = []
        self.errors_encountered = []

        # Set epochs based on mode
        if quick_mode:
            self.epochs = 5
            self.mode_name = "Quick Demo"
        elif full_mode:
            self.epochs = 50
            self.mode_name = "Full Demo"
        else:
            self.epochs = 10
            self.mode_name = "Standard Demo"

        # Setup logging
        self.setup_logging()

        # Validate environment
        self.validate_environment()

    def setup_logging(self):
        """Setup comprehensive logging for demo execution."""
        # Create logs directory
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"demo_{self.ticker}_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

        self.logger.info(f"Demo started: {self.mode_name} for {self.ticker}")
        self.logger.info(f"Log file: {log_file}")

    def validate_environment(self):
        """Validate the environment and detect system capabilities."""
        self.logger.info("Validating environment...")

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            raise RuntimeError(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")

        # Check for GPU availability
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"[OK] GPU detected: {gpu_name}")
            else:
                self.logger.info("[OK] Using CPU (GPU not available)")
        except ImportError:
            self.gpu_available = False
            self.logger.warning("PyTorch not installed - will attempt installation")

        # Check required directories
        required_dirs = ["data", "models", "scalers", "predictions", "results"]
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            dir_path.mkdir(exist_ok=True)
            self.logger.info(f"[OK] Directory ready: {dir_name}/")

        self.logger.info("Environment validation complete")

    def run_command(self, cmd: str, description: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """
        Execute a command with comprehensive error handling and monitoring.

        Args:
            cmd: Command to execute
            description: Human-readable description
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"STEP: {description}")
        self.logger.info(f"COMMAND: {cmd}")
        self.logger.info(f"{'='*80}")

        try:
            # Execute command with timeout
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT
            )

            # Log output
            if result.stdout:
                self.logger.info("STDOUT:")
                self.logger.info(result.stdout)

            if result.stderr:
                self.logger.warning("STDERR:")
                self.logger.warning(result.stderr)

            # Check return code
            if result.returncode == 0:
                self.logger.info(f"[OK] {description} completed successfully")
                self.steps_completed.append(description)
                return True, result.stdout, result.stderr
            else:
                self.logger.error(f"[FAIL] {description} failed with return code {result.returncode}")
                self.errors_encountered.append(f"{description}: Return code {result.returncode}")
                return False, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            self.logger.error(f"[FAIL] {description} failed: {error_msg}")
            self.errors_encountered.append(f"{description}: {error_msg}")
            return False, "", error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"[FAIL] {description} failed: {error_msg}")
            self.errors_encountered.append(f"{description}: {error_msg}")
            return False, "", error_msg

    def check_file_exists(self, file_path: Path, description: str) -> bool:
        """Check if a file exists and log the result."""
        if file_path.exists():
            size = file_path.stat().st_size
            self.logger.info(f"[OK] {description}: {file_path} ({size:,} bytes)")
            return True
        else:
            self.logger.error(f"[FAIL] {description}: {file_path} not found")
            return False

    def step_1_download_data(self) -> bool:
        """Step 1: Download historical stock data."""
        self.logger.info("\n[STEP 1] Downloading stock data...")

        success, stdout, stderr = self.run_command(
            f"python scripts/data/download_data.py --ticker {self.ticker}",
            f"Download {self.ticker} stock data",
            timeout=60
        )

        if success:
            # Verify data file was created
            data_file = PROJECT_ROOT / "data" / "raw" / f"{self.ticker}.csv"
            if self.check_file_exists(data_file, f"{self.ticker} data file"):
                # Quick data validation
                try:
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    self.logger.info(f"[OK] Data validation: {len(df)} rows, {len(df.columns)} columns")
                    self.logger.info(f"[OK] Date range: {df.index[0] if df.index.name else 'N/A'} to {df.index[-1] if df.index.name else 'N/A'}")
                except Exception as e:
                    self.logger.warning(f"Data validation warning: {e}")

                return True

        self.logger.error("Data download failed - check network connection and ticker symbol")
        return False

    def step_2_train_model(self) -> bool:
        """Step 2: Train the transformer model."""
        self.logger.info(f"\n[STEP 2] Training model ({self.epochs} epochs)...")

        # Construct training command with appropriate settings
        cmd_parts = [
            "python scripts/training/train_ultra_simple.py",
            f"--ticker {self.ticker}",
            f"--epochs {self.epochs}",
            "--batch-size 32",  # Conservative batch size for compatibility
        ]

        # Add GPU/CPU specific settings
        if not self.gpu_available:
            cmd_parts.append("--device cpu")

        cmd = " ".join(cmd_parts)

        # Set appropriate timeout based on epochs
        timeout = 60 + (self.epochs * 30)  # Base 60s + 30s per epoch

        success, stdout, stderr = self.run_command(
            cmd,
            f"Train {self.ticker} model",
            timeout=timeout
        )

        if success:
            # Verify model files were created
            model_file = PROJECT_ROOT / "models" / f"model_{self.ticker}_best.pt"
            scaler_file = PROJECT_ROOT / "scalers" / f"scaler_{self.ticker}.json"

            model_exists = self.check_file_exists(model_file, f"{self.ticker} model file")
            scaler_exists = self.check_file_exists(scaler_file, f"{self.ticker} scaler file")

            if model_exists and scaler_exists:
                self.logger.info(f"[OK] Model training completed successfully")
                return True

        self.logger.error("Model training failed - check system resources and data quality")
        return False

    def step_3_generate_predictions(self) -> bool:
        """Step 3: Generate predictions using the trained model."""
        self.logger.info("\n[STEP 3] Generating predictions...")

        # Use the bias-adjusted prediction script with ticker-based model
        success, stdout, stderr = self.run_command(
            f"python scripts/bias_adjusted_predictions.py --ticker {self.ticker} --model-path models/model_{self.ticker}_best.pt",
            f"Generate {self.ticker} predictions",
            timeout=120
        )

        if success:
            # Check for prediction outputs - try ticker-specific first
            predictions_dir = PROJECT_ROOT / "predictions"
            prediction_files = list(predictions_dir.glob(f"prediction_{self.ticker}_*.json"))

            if not prediction_files:
                # Fallback to general predictions
                prediction_files = list(predictions_dir.glob("predictions_*.json"))

            if prediction_files:
                latest_predictions = max(prediction_files, key=lambda x: x.stat().st_mtime)
                self.check_file_exists(latest_predictions, "Latest predictions file")

                # Quick validation of predictions
                try:
                    with open(latest_predictions, 'r') as f:
                        predictions_data = json.load(f)

                    if isinstance(predictions_data, dict) and len(predictions_data) > 0:
                        self.logger.info(f"[OK] Predictions generated: {len(predictions_data)} predictions")
                        return True
                except Exception as e:
                    self.logger.warning(f"Prediction validation warning: {e}")

            # Alternative: check if stdout contains prediction information
            if "prediction" in stdout.lower() or "signal" in stdout.lower():
                self.logger.info("[OK] Predictions generated successfully")
                return True

        self.logger.error("Prediction generation failed")
        return False

    def step_4_run_backtest(self) -> bool:
        """Step 4: Run backtesting analysis (optional)."""
        self.logger.info("\n[STEP 4] Running backtesting analysis...")

        # Check if we have the required files for backtesting
        data_file = PROJECT_ROOT / "data" / "raw" / f"{self.ticker}.csv"
        model_file = PROJECT_ROOT / "models" / f"model_{self.ticker}_best.pt"

        if not (data_file.exists() and model_file.exists()):
            self.logger.warning("Skipping backtest - missing required files")
            return True  # Not a failure, just skipped

        # First check if a new predictions file was just created
        import glob
        predictions_dir = PROJECT_ROOT / "predictions"
        prediction_files = glob.glob(str(predictions_dir / f"prediction_{self.ticker}_*.json"))
        if prediction_files:
            latest_predictions = Path(max(prediction_files, key=os.path.getmtime))
        else:
            # Fallback to any predictions file
            prediction_files = list(predictions_dir.glob("predictions_*.json"))
            if not prediction_files:
                self.logger.warning("Skipping backtest - no predictions available")
                return True
            latest_predictions = max(prediction_files, key=lambda x: x.stat().st_mtime)

        # Run a simple backtest
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # 3 months

        cmd = (
            f"python scripts/backtesting/run_backtest.py "
            f"--predictions-path {latest_predictions} "
            f"--market-data-path {data_file} "
            f"--start-date {start_date} "
            f"--end-date {end_date} "
            f"--initial-capital 100000"
        )

        success, stdout, stderr = self.run_command(
            cmd,
            f"Run {self.ticker} backtest",
            timeout=180
        )

        if success:
            # Check for backtest results
            results_dir = PROJECT_ROOT / "results" / "backtest"
            if results_dir.exists():
                result_files = list(results_dir.glob("backtest_report_*.json"))
                if result_files:
                    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                    self.check_file_exists(latest_result, "Backtest results")
                    return True

        self.logger.warning("Backtest completed with warnings - check results manually")
        return True  # Non-critical failure

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        self.logger.info("\n[REPORT] Generating summary report...")

        end_time = datetime.now()
        duration = end_time - self.start_time

        # Collect file information
        files_created = []

        # Check for key output files
        key_files = [
            (PROJECT_ROOT / "data" / "raw" / f"{self.ticker}.csv", "Stock data"),
            (PROJECT_ROOT / "models" / f"model_{self.ticker}_best.pt", "Trained model"),
            (PROJECT_ROOT / "scalers" / f"scaler_{self.ticker}.json", "Scaler parameters"),
        ]

        # Check for prediction files
        predictions_dir = PROJECT_ROOT / "predictions"
        if predictions_dir.exists():
            prediction_files = list(predictions_dir.glob("predictions_*.json"))
            if prediction_files:
                latest_predictions = max(prediction_files, key=lambda x: x.stat().st_mtime)
                key_files.append((latest_predictions, "Latest predictions"))

        # Check for backtest results
        results_dir = PROJECT_ROOT / "results" / "backtest"
        if results_dir.exists():
            result_files = list(results_dir.glob("backtest_report_*.json"))
            if result_files:
                latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                key_files.append((latest_result, "Backtest results"))

        # Generate report
        for file_path, description in key_files:
            if file_path.exists():
                size = file_path.stat().st_size
                files_created.append(f"  [OK] {description}: {file_path.name} ({size:,} bytes)")
            else:
                files_created.append(f"  [FAIL] {description}: Not created")

        # Create summary
        summary = {
            "demo_info": {
                "ticker": self.ticker,
                "mode": self.mode_name,
                "epochs": self.epochs,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "log_file": str(self.log_file)
            },
            "execution_summary": {
                "steps_completed": len(self.steps_completed),
                "steps_failed": len(self.errors_encountered),
                "success_rate": len(self.steps_completed) / (len(self.steps_completed) + len(self.errors_encountered)) if (len(self.steps_completed) + len(self.errors_encountered)) > 0 else 0
            },
            "steps_completed": self.steps_completed,
            "errors_encountered": self.errors_encountered,
            "files_created": files_created
        }

        # Save summary report
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        summary_file = PROJECT_ROOT / "results" / f"demo_summary_{self.ticker}_{timestamp}.json"
        summary_file.parent.mkdir(exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary report saved: {summary_file}")
        return summary

    def print_final_report(self, summary: Dict):
        """Print a user-friendly final report."""
        print("\n" + "="*80)
        print(f"[DEMO COMPLETE] {self.ticker} {self.mode_name}")
        print("="*80)

        duration = summary["demo_info"]["duration_seconds"]
        print(f"Duration: {duration//60:.0f}m {duration%60:.0f}s")
        print(f"Steps completed: {summary['execution_summary']['steps_completed']}")

        if summary['execution_summary']['steps_failed'] > 0:
            print(f"Steps failed: {summary['execution_summary']['steps_failed']}")

        print(f"Success rate: {summary['execution_summary']['success_rate']:.1%}")

        print("\nFiles Created:")
        for file_info in summary["files_created"]:
            print(file_info)

        print(f"\nFull log: {summary['demo_info']['log_file']}")

        # Next steps guidance
        print("\n" + "="*80)
        print("NEXT STEPS TO EXTEND THIS PROJECT")
        print("="*80)

        print("\n1. Extend with additional data:")
        print("   python scripts/run_demo.py --ticker MSFT")
        print("   python scripts/run_demo.py --ticker NVDA --full")

        print("\n2. Customize model parameters:")
        print("   - Edit configs/transformer_base.yaml")
        print("   - Modify training parameters in train_ultra_simple.py")

        print("\n3. Advanced analysis:")
        print("   python scripts/backtesting/run_backtest.py --walk-forward")
        print("   python scripts/evaluation/evaluate.py")

        print("\n4. Automate daily runs:")
        print("   - Set up cron job (Linux/Mac) or Task Scheduler (Windows)")
        print("   - Use scripts/run_demo.py in automated workflows")

        print("\n5. Experiment with strategies:")
        print("   - Implement new prediction heads in src/models/components/")
        print("   - Test different risk management parameters")

        print("\n6. Production deployment:")
        print("   - Containerize with Docker")
        print("   - Set up monitoring and alerting")
        print("   - Implement real-time data feeds")

        if summary['execution_summary']['success_rate'] < 1.0:
            print("\nWARNING: Some steps failed. Check the log file for details and troubleshooting guidance.")

    def run(self) -> bool:
        """Execute the complete demo pipeline."""
        self.logger.info(f"Starting {self.mode_name} for {self.ticker}")

        try:
            # Execute pipeline steps
            steps = [
                ("Download Data", self.step_1_download_data),
                ("Train Model", self.step_2_train_model),
                ("Generate Predictions", self.step_3_generate_predictions),
                ("Run Backtest", self.step_4_run_backtest),
            ]

            success_count = 0
            for step_name, step_func in steps:
                try:
                    if step_func():
                        success_count += 1
                    else:
                        self.logger.error(f"Step failed: {step_name}")
                except Exception as e:
                    self.logger.error(f"Step error: {step_name} - {e}")
                    self.errors_encountered.append(f"{step_name}: {str(e)}")

            # Generate reports
            summary = self.generate_summary_report()
            self.print_final_report(summary)

            # Return overall success
            overall_success = success_count >= 3  # Allow backtest to be optional

            if overall_success:
                print(f"\nDemo completed successfully!")
            else:
                print(f"\nWARNING: Demo completed with issues. Check logs for details.")

            return overall_success

        except Exception as e:
            self.logger.error(f"Critical error in demo execution: {e}")
            print(f"\nERROR: Demo failed: {e}")
            return False


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="TimeSeries Transformer Complete Demo - Production-ready ML trading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard demo with AAPL (10 epochs)
  python scripts/run_demo.py

  # Run quick demo with different ticker
  python scripts/run_demo.py --ticker MSFT --quick

  # Run full demo with comprehensive training
  python scripts/run_demo.py --ticker NVDA --full

  # Run with specific ticker
  python scripts/run_demo.py --ticker TSLA

This script will:
1. Download historical stock data
2. Train a transformer model with GPU/CPU detection
3. Generate bias-adjusted predictions
4. Run backtesting analysis
5. Provide comprehensive results and next steps

All outputs are saved with timestamps for easy tracking.
"""
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol to use for demo. Default: AAPL. Popular choices: MSFT, NVDA, TSLA, GOOGL"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo mode (5 epochs) - faster execution for testing"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full demo mode (50 epochs) - comprehensive training for better results"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.quick and args.full:
        print("Error: Cannot specify both --quick and --full modes")
        return 1

    # Create and run demo
    try:
        demo = DemoRunner(
            ticker=args.ticker,
            quick_mode=args.quick,
            full_mode=args.full
        )

        success = demo.run()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Demo failed with critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())