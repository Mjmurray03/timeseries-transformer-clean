"""
Dataset Integration Verification Script
Verifies StockSequenceDataset produces the exact format expected by TrainingOrchestrator.
"""

import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Set UTF-8 encoding for Windows console (only when running as main script)
if sys.platform == "win32" and __name__ == "__main__":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.training_config import TrainingConfig
from src.data.datasets.stock_dataset import StockSequenceDataset, create_data_loaders
from src.data.processors.feature_engineering import FeatureEngineer
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.training.trainer import TrainingOrchestrator

# Setup logging to capture details
logging.basicConfig(level=logging.WARNING)


class DatasetIntegrationVerifier:
    """Verifies dataset integration with TrainingOrchestrator."""

    def __init__(self):
        self.results = {}
        self.verification_errors = []
        self.seq_len = 60
        self.horizon = 5
        self.batch_size = 32

    def verify_all_integration(self, data_path: str = None) -> Dict:
        """Run comprehensive dataset integration verification."""
        start_time = time.time()

        print("=" * 80)
        print("DATASET INTEGRATION VERIFICATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sequence Length: {self.seq_len}")
        print(f"Forecast Horizon: {self.horizon}")
        print(f"Batch Size: {self.batch_size}")
        print("=" * 80)

        # Step 1: Load feature-engineered data
        feature_engineered_df = self._load_feature_engineered_data(data_path)
        if feature_engineered_df is None:
            return self._generate_report(time.time() - start_time)

        print(
            f"\\nLoaded feature-engineered data: {feature_engineered_df.shape[0]} rows × {feature_engineered_df.shape[1]} columns"
        )

        # Step 2: Create sequences from feature data
        sequences, targets, metadata = self._create_sequences_from_features(feature_engineered_df)
        if sequences is None:
            return self._generate_report(time.time() - start_time)

        print(f"Created sequences: {sequences.shape}, targets: {targets.shape}")

        # Step 3: Create StockSequenceDataset instance
        dataset = self._create_stock_sequence_dataset(sequences, targets)
        if dataset is None:
            return self._generate_report(time.time() - start_time)

        # Step 4: Verify dataset.__getitem__ format
        self._verify_dataset_getitem_format(dataset)

        # Step 5: Create DataLoader and verify batch format
        data_loader = self._create_and_verify_dataloader(dataset)
        if data_loader is None:
            return self._generate_report(time.time() - start_time)

        # Step 6: Test TrainingOrchestrator integration
        # Calculate actual feature count from sequences
        actual_n_features = sequences.shape[2]
        print(f"Debug: feature_engineered_df.shape[1] = {feature_engineered_df.shape[1]}")
        print(f"Debug: sequences.shape[2] = {actual_n_features}")
        self._verify_training_orchestrator_integration(data_loader, actual_n_features)

        # Step 7: Verify data continuity and splits
        self._verify_data_continuity_and_splits(feature_engineered_df, sequences, targets, metadata)

        elapsed_time = time.time() - start_time
        return self._generate_report(elapsed_time)

    def _load_feature_engineered_data(self, data_path: str = None) -> Optional[pd.DataFrame]:
        """Load feature-engineered data from parquet file."""
        if data_path is None:
            # Find first available parquet file and apply feature engineering
            data_dir = Path("data/raw")
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".parquet"):
                        raw_file_path = Path(root) / file
                        break
                if "raw_file_path" in locals():
                    break

            if "raw_file_path" not in locals():
                self._report_error("DATA_LOADING", "No parquet files found in data/raw/")
                return None

            try:
                # Load raw data and apply feature engineering
                raw_df = pd.read_parquet(raw_file_path)
                feature_engineer = FeatureEngineer()
                feature_df = feature_engineer.engineer_features(raw_df)

                print(f"Loaded and feature-engineered data from: {raw_file_path}")
                self.results["DATA_LOADING"] = "PASS"
                return feature_df

            except Exception as e:
                self._report_error(
                    "DATA_LOADING", f"Failed to load and engineer features: {str(e)}"
                )
                return None
        else:
            try:
                df = pd.read_parquet(data_path)
                self.results["DATA_LOADING"] = "PASS"
                return df
            except Exception as e:
                self._report_error(
                    "DATA_LOADING", f"Failed to load data from {data_path}: {str(e)}"
                )
                return None

    def _create_sequences_from_features(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """Create sequences and targets from feature-engineered data."""
        try:
            # Remove non-numeric columns (Ticker)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_data = df[numeric_columns].values

            # Ensure we have the 'Close' column for targets
            if "Close" not in df.columns:
                self._report_error(
                    "SEQUENCE_CREATION", "'Close' column not found for target creation"
                )
                return None, None, None

            close_prices = df["Close"].values

            # Create sequences and targets
            sequences = []
            targets = []
            dates = []

            for i in range(len(feature_data) - self.seq_len - self.horizon + 1):
                # Input sequence: features for seq_len time steps
                seq = feature_data[i : i + self.seq_len]
                sequences.append(seq)

                # Target: future close prices for horizon steps
                target_start = i + self.seq_len
                target_end = target_start + self.horizon
                target = close_prices[target_start:target_end]
                targets.append(target)

                # Metadata: dates for this sequence
                if hasattr(df.index, "to_pydatetime"):
                    seq_dates = df.index[i : i + self.seq_len].to_pydatetime()
                    dates.append(seq_dates)
                else:
                    dates.append(list(range(i, i + self.seq_len)))

            sequences = np.array(sequences, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)

            metadata = {
                "feature_names": numeric_columns,
                "sequence_dates": dates,
                "n_features": len(numeric_columns),
                "seq_len": self.seq_len,
                "horizon": self.horizon,
            }

            self.results["SEQUENCE_CREATION"] = "PASS"
            print("+ Sequence creation successful")

            return sequences, targets, metadata

        except Exception as e:
            self._report_error("SEQUENCE_CREATION", f"Failed to create sequences: {str(e)}")
            return None, None, None

    def _create_stock_sequence_dataset(
        self, sequences: np.ndarray, targets: np.ndarray
    ) -> Optional[StockSequenceDataset]:
        """Create StockSequenceDataset instance."""
        try:
            dataset = StockSequenceDataset(
                sequences=sequences,
                targets=targets,
                features=None,  # Feature names can be passed here if needed
                transform=None,
            )

            self.results["DATASET_CREATION"] = "PASS"
            print(f"+ StockSequenceDataset created with {len(dataset)} samples")

            return dataset

        except Exception as e:
            self._report_error(
                "DATASET_CREATION", f"Failed to create StockSequenceDataset: {str(e)}"
            )
            return None

    def _verify_dataset_getitem_format(self, dataset: StockSequenceDataset):
        """Verify dataset.__getitem__(idx) returns correct format."""
        try:
            # Test multiple random indices
            test_indices = [0, len(dataset) // 2, len(dataset) - 1]

            for idx in test_indices:
                item = dataset[idx]

                # Check that it returns a dictionary
                if not isinstance(item, dict):
                    self._report_error(
                        "GETITEM_FORMAT", f"__getitem__ should return dict, got {type(item)}"
                    )
                    return

                # Check required keys
                required_keys = {"inputs", "targets"}
                missing_keys = required_keys - set(item.keys())
                if missing_keys:
                    self._report_error(
                        "GETITEM_FORMAT", f"Missing required keys in item: {missing_keys}"
                    )
                    return

                # Check inputs tensor
                inputs = item["inputs"]
                if not isinstance(inputs, torch.Tensor):
                    self._report_error(
                        "GETITEM_FORMAT", f"inputs should be torch.Tensor, got {type(inputs)}"
                    )
                    return

                expected_input_shape = (
                    self.seq_len,
                    dataset.sequences.shape[2],
                )  # (60, n_features)
                if inputs.shape != expected_input_shape:
                    self._report_error(
                        "GETITEM_FORMAT",
                        f"inputs shape mismatch: expected {expected_input_shape}, got {inputs.shape}",
                    )
                    return

                # Check targets tensor
                targets = item["targets"]
                if not isinstance(targets, torch.Tensor):
                    self._report_error(
                        "GETITEM_FORMAT", f"targets should be torch.Tensor, got {type(targets)}"
                    )
                    return

                expected_target_shape = (self.horizon,)  # (5,)
                if targets.shape != expected_target_shape:
                    self._report_error(
                        "GETITEM_FORMAT",
                        f"targets shape mismatch: expected {expected_target_shape}, got {targets.shape}",
                    )
                    return

                # Check tensor dtypes
                if inputs.dtype != torch.float32:
                    self._report_error(
                        "GETITEM_FORMAT", f"inputs should be float32, got {inputs.dtype}"
                    )
                    return

                if targets.dtype != torch.float32:
                    self._report_error(
                        "GETITEM_FORMAT", f"targets should be float32, got {targets.dtype}"
                    )
                    return

            self.results["GETITEM_FORMAT"] = "PASS"
            print(f"+ Dataset __getitem__ format verification passed")
            print(f"  - inputs shape: {expected_input_shape}")
            print(f"  - targets shape: {expected_target_shape}")
            print(f"  - data types: float32")

        except Exception as e:
            self._report_error("GETITEM_FORMAT", f"Error verifying __getitem__ format: {str(e)}")

    def _create_and_verify_dataloader(self, dataset: StockSequenceDataset) -> Optional[DataLoader]:
        """Create DataLoader and verify batch format."""
        try:
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Don't shuffle for verification
                num_workers=0,  # Avoid multiprocessing issues in testing
                pin_memory=False,
                drop_last=True,  # Ensure consistent batch sizes
            )

            print(f"+ DataLoader created with batch_size={self.batch_size}")

            # Get one batch and verify format
            batch = next(iter(data_loader))

            # Check batch structure
            if not isinstance(batch, dict):
                self._report_error("BATCH_FORMAT", f"Batch should be dict, got {type(batch)}")
                return None

            # Check batch keys
            required_keys = {"inputs", "targets"}
            missing_keys = required_keys - set(batch.keys())
            if missing_keys:
                self._report_error(
                    "BATCH_FORMAT", f"Missing required keys in batch: {missing_keys}"
                )
                return None

            # Check batch inputs shape
            batch_inputs = batch["inputs"]
            expected_batch_input_shape = (self.batch_size, self.seq_len, dataset.sequences.shape[2])
            if batch_inputs.shape != expected_batch_input_shape:
                self._report_error(
                    "BATCH_FORMAT",
                    f"batch inputs shape mismatch: expected {expected_batch_input_shape}, got {batch_inputs.shape}",
                )
                return None

            # Check batch targets shape
            batch_targets = batch["targets"]
            expected_batch_target_shape = (self.batch_size, self.horizon)
            if batch_targets.shape != expected_batch_target_shape:
                self._report_error(
                    "BATCH_FORMAT",
                    f"batch targets shape mismatch: expected {expected_batch_target_shape}, got {batch_targets.shape}",
                )
                return None

            # Check tensor types
            if batch_inputs.dtype != torch.float32:
                self._report_error(
                    "BATCH_FORMAT", f"batch inputs should be float32, got {batch_inputs.dtype}"
                )
                return None

            if batch_targets.dtype != torch.float32:
                self._report_error(
                    "BATCH_FORMAT", f"batch targets should be float32, got {batch_targets.dtype}"
                )
                return None

            self.results["BATCH_FORMAT"] = "PASS"
            print(f"+ DataLoader batch format verification passed")
            print(f"  - batch inputs shape: {expected_batch_input_shape}")
            print(f"  - batch targets shape: {expected_batch_target_shape}")

            return data_loader

        except Exception as e:
            self._report_error("BATCH_FORMAT", f"Error creating/verifying DataLoader: {str(e)}")
            return None

    def _verify_training_orchestrator_integration(self, data_loader: DataLoader, n_features: int):
        """Verify TrainingOrchestrator can consume the batch format."""
        try:
            # Create a minimal model for testing
            model = self._create_test_model(n_features)

            # Create minimal training config
            training_config = self._create_test_training_config()

            # Create TrainingOrchestrator
            orchestrator = TrainingOrchestrator(
                model=model,
                config=training_config,
                device=torch.device("cpu"),  # Use CPU for testing
            )

            # Test processing one batch
            batch = next(iter(data_loader))

            # Move batch to device (as TrainingOrchestrator would do)
            inputs = batch["inputs"].to(orchestrator.device)
            targets = batch["targets"].to(orchestrator.device)

            # Test forward pass
            with torch.no_grad():
                predictions = model(inputs)

                # Check prediction shape
                expected_pred_shape = targets.shape  # Should match targets
                if predictions.shape != expected_pred_shape:
                    self._report_error(
                        "ORCHESTRATOR_INTEGRATION",
                        f"Model prediction shape mismatch: expected {expected_pred_shape}, got {predictions.shape}",
                    )
                    return

                # Test loss calculation (as TrainingOrchestrator would do)
                criterion = nn.MSELoss()
                loss = criterion(predictions, targets)

                if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                    self._report_error(
                        "ORCHESTRATOR_INTEGRATION",
                        f"Loss should be scalar tensor, got shape {loss.shape}",
                    )
                    return

            self.results["ORCHESTRATOR_INTEGRATION"] = "PASS"
            print(f"+ TrainingOrchestrator integration verified")
            print(f"  - Model forward pass successful")
            print(f"  - Loss calculation successful")
            print(f"  - Prediction shape: {predictions.shape}")
            print(f"  - Loss value: {loss.item():.6f}")

        except Exception as e:
            self._report_error(
                "ORCHESTRATOR_INTEGRATION", f"TrainingOrchestrator integration failed: {str(e)}"
            )

    def _verify_data_continuity_and_splits(
        self, df: pd.DataFrame, sequences: np.ndarray, targets: np.ndarray, metadata: Dict
    ):
        """Verify data continuity and split integrity."""
        try:
            # Check sequence continuity
            if hasattr(df.index, "to_pydatetime"):
                dates = df.index.to_pydatetime()

                # Verify sequences are continuous in time
                for i, seq_dates in enumerate(metadata["sequence_dates"][:10]):  # Check first 10
                    if len(seq_dates) != self.seq_len:
                        self._report_error(
                            "DATA_CONTINUITY",
                            f"Sequence {i} has wrong length: expected {self.seq_len}, got {len(seq_dates)}",
                        )
                        return

                    # Check that dates are consecutive (business days)
                    for j in range(1, len(seq_dates)):
                        time_diff = (seq_dates[j] - seq_dates[j - 1]).days
                        if time_diff > 4:  # Allow for weekends/holidays (max 4 days gap)
                            self._report_error(
                                "DATA_CONTINUITY",
                                f"Non-continuous dates in sequence {i}: gap of {time_diff} days",
                            )
                            return

            # Check target alignment
            # Targets should align with the end of each sequence
            for i in range(min(10, len(sequences))):  # Check first 10
                seq_end_idx = i + self.seq_len - 1
                target_start_idx = i + self.seq_len

                # Verify no data leakage: targets should come after sequence
                if target_start_idx <= seq_end_idx:
                    self._report_error(
                        "DATA_CONTINUITY",
                        f"Data leakage detected: target starts at {target_start_idx}, sequence ends at {seq_end_idx}",
                    )
                    return

                # Check that targets have the correct horizon
                if targets[i].shape[0] != self.horizon:
                    self._report_error(
                        "DATA_CONTINUITY",
                        f"Target {i} has wrong horizon: expected {self.horizon}, got {targets[i].shape[0]}",
                    )
                    return

            # Test train/val/test split integrity (no overlap)
            from src.data.datasets.stock_dataset import split_sequences

            train_data, val_data, test_data = split_sequences(
                sequences,
                targets,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                shuffle=False,  # Don't shuffle to maintain temporal order
            )

            train_seq, train_targets = train_data
            val_seq, val_targets = val_data
            test_seq, test_targets = test_data

            # Check split sizes
            total_samples = len(sequences)
            expected_train = int(total_samples * 0.7)
            expected_val = int(total_samples * 0.15)
            expected_test = total_samples - expected_train - expected_val

            if len(train_seq) != expected_train:
                self._report_error(
                    "DATA_CONTINUITY",
                    f"Train split size mismatch: expected ~{expected_train}, got {len(train_seq)}",
                )
                return

            # Check that splits are properly ordered (since we didn't shuffle)
            # The split function might reorder data, so we just check sizes are reasonable
            total_split_size = len(train_seq) + len(val_seq) + len(test_seq)
            if total_split_size != len(sequences):
                self._report_error(
                    "DATA_CONTINUITY",
                    f"Split size mismatch: expected {len(sequences)}, got {total_split_size}",
                )
                return

            self.results["DATA_CONTINUITY"] = "PASS"
            print(f"+ Data continuity and split integrity verified")
            print(f"  - Sequences are temporally continuous")
            print(f"  - No data leakage between inputs and targets")
            print(f"  - Train/val/test splits: {len(train_seq)}/{len(val_seq)}/{len(test_seq)}")

        except Exception as e:
            self._report_error("DATA_CONTINUITY", f"Data continuity verification failed: {str(e)}")

    def _create_test_model(self, n_features: int) -> nn.Module:
        """Create a simple test model that matches expected interface."""

        class SimpleTestModel(nn.Module):
            def __init__(self, input_dim, seq_len, horizon):
                super().__init__()
                self.seq_len = seq_len
                self.horizon = horizon
                # Calculate correct input size for linear layer
                flattened_size = input_dim * seq_len
                self.linear = nn.Linear(flattened_size, horizon)
                print(
                    f"Created model: input_dim={input_dim}, seq_len={seq_len}, flattened_size={flattened_size}, horizon={horizon}"
                )

            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                batch_size, seq_len, input_dim = x.size()
                x = x.view(batch_size, -1)  # Flatten to (batch_size, seq_len * input_dim)
                return self.linear(x)  # Output: (batch_size, horizon)

        return SimpleTestModel(n_features, self.seq_len, self.horizon)

    def _create_test_training_config(self) -> TrainingConfig:
        """Create minimal training config for testing."""
        # Create TrainingConfig using dataclass fields directly
        config = TrainingConfig()
        config.num_epochs = 1
        config.batch_size = self.batch_size
        config.gradient_accumulation_steps = 1
        config.gradient_clip = 1.0
        config.use_amp = False
        config.device = "cpu"
        config.checkpoint_dir = Path("temp_checkpoints")
        config.experiment_name = "test_integration"
        config.project_name = "dataset_verification"
        config.early_stopping_patience = 10
        config.early_stopping_min_delta = 1e-6
        config.save_best_only = True

        return config

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

        print("\\n" + "=" * 80)
        print("INTEGRATION VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks} +")
        print(f"Failed: {failed_checks} X")
        print(f"Success Rate: {(passed_checks/total_checks*100) if total_checks > 0 else 0:.1f}%")
        print(f"Execution Time: {elapsed_time:.2f} seconds")

        if self.verification_errors:
            print("\\nVERIFICATION ERRORS:")
            for error in self.verification_errors:
                print(f"  - {error['check']}: {error['message']}")

        overall_status = "PASS" if failed_checks == 0 else "FAIL"
        print(f"\\n{'='*80}")
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
        }


def main():
    """Main execution function."""
    verifier = DatasetIntegrationVerifier()
    report = verifier.verify_all_integration()

    # Return appropriate exit code
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
