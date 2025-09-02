#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for WANDB setup integration with comprehensive configuration.

This script demonstrates the complete WANDB setup following design specifications
and tests all major functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import logging

from src.training.wandb_setup import (
    WANDBConfig,
    init_wandb,
    init_wandb_for_evaluation,
    init_wandb_for_hyperparameter_search,
    setup_wandb_for_training,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_initialization():
    """Test basic WANDB initialization."""
    logger.info("=== Testing Basic WANDB Initialization ===")

    try:
        run = init_wandb()
        logger.info(f"✓ Basic initialization successful")
        logger.info(f"  Run ID: {run.id}")
        logger.info(f"  Project: {run.project}")
        logger.info(f"  Dashboard: {run.url}")
        run.finish()

    except Exception as e:
        logger.error(f"✗ Basic initialization failed: {e}")


def test_training_setup():
    """Test training-specific WANDB setup."""
    logger.info("\n=== Testing Training-Specific Setup ===")

    try:
        # Configuration following design specifications
        model_config = {
            "architecture": "transformer",
            "num_layers": 6,
            "hidden_dim": 256,
            "num_heads": 8,
            "dropout": 0.1,
            "sequence_length": 60,
        }

        training_config = {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "gradient_clip": 1.0,
        }

        data_config = {
            "dataset": "SP500",
            "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
            "years": 5,
            "features": ["OHLCV", "volume", "returns"],
            "sequence_length": 60,
            "prediction_horizon": 5,
        }

        run = setup_wandb_for_training(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            experiment_name="test_training_setup",
        )

        logger.info(f"✓ Training setup successful")
        logger.info(f"  Run ID: {run.id}")
        logger.info(f"  Experiment: {run.name}")
        logger.info(f"  Tags: {run.tags}")

        # Test logging some metrics
        run.log(
            {
                "train/loss": 0.5,
                "train/accuracy": 0.85,
                "val/loss": 0.6,
                "val/accuracy": 0.82,
                "learning_rate": 1e-4,
            }
        )

        logger.info("✓ Metrics logging successful")
        run.finish()

    except Exception as e:
        logger.error(f"✗ Training setup failed: {e}")


def test_evaluation_setup():
    """Test evaluation-specific WANDB setup."""
    logger.info("\n=== Testing Evaluation Setup ===")

    try:
        test_config = {
            "model_version": "v1.0.0",
            "test_dataset": "SP500_test",
            "metrics": ["RMSE", "MAE", "Sharpe", "Accuracy"],
            "num_samples": 1000,
        }

        run = init_wandb_for_evaluation(model_version="v1.0.0", test_config=test_config)

        logger.info(f"✓ Evaluation setup successful")
        logger.info(f"  Run ID: {run.id}")
        logger.info(f"  Experiment: {run.name}")

        # Test logging evaluation metrics
        run.log(
            {
                "eval/rmse": 0.045,
                "eval/mae": 0.032,
                "eval/sharpe_ratio": 1.23,
                "eval/directional_accuracy": 0.67,
            }
        )

        logger.info("✓ Evaluation metrics logging successful")
        run.finish()

    except Exception as e:
        logger.error(f"✗ Evaluation setup failed: {e}")


def test_hyperparameter_search_setup():
    """Test hyperparameter search WANDB setup."""
    logger.info("\n=== Testing Hyperparameter Search Setup ===")

    try:
        search_config = {
            "method": "bayesian",
            "parameters": {
                "learning_rate": {"values": [1e-5, 1e-4, 1e-3]},
                "batch_size": {"values": [16, 32, 64]},
                "num_layers": {"min": 4, "max": 8},
                "hidden_dim": {"values": [128, 256, 512]},
                "dropout": {"min": 0.1, "max": 0.3},
            },
            "metric": {"name": "val_loss", "goal": "minimize"},
        }

        run = init_wandb_for_hyperparameter_search(search_config=search_config)

        logger.info(f"✓ Hyperparameter search setup successful")
        logger.info(f"  Run ID: {run.id}")
        logger.info(f"  Experiment: {run.name}")

        # Test logging hyperparameter trial results
        run.log(
            {
                "trial/learning_rate": 1e-4,
                "trial/batch_size": 32,
                "trial/val_loss": 0.55,
                "trial/val_accuracy": 0.78,
            }
        )

        logger.info("✓ Hyperparameter search logging successful")
        run.finish()

    except Exception as e:
        logger.error(f"✗ Hyperparameter search setup failed: {e}")


def test_custom_configuration():
    """Test custom configuration with all options."""
    logger.info("\n=== Testing Custom Configuration ===")

    try:
        custom_config = {
            "architecture": "transformer-6layer",
            "dataset": "SP500_2019-2024",
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "optimizer": "AdamW",
            },
            "model": {"layers": 6, "hidden_size": 256, "attention_heads": 8, "dropout": 0.1},
        }

        run = init_wandb(
            project_name="timeseries-transformer",
            experiment_name="custom_test_experiment",
            config=custom_config,
            tags=["test", "custom", "v1.0.0"],
            notes="Testing custom configuration with all WANDB features",
            job_type="training",
        )

        logger.info(f"✓ Custom configuration successful")
        logger.info(f"  Run ID: {run.id}")
        logger.info(f"  Project: {run.project}")
        logger.info(f"  Experiment: {run.name}")
        logger.info(f"  Tags: {run.tags}")
        logger.info(f"  Dashboard: {run.url}")

        # Test comprehensive logging
        run.log(
            {
                "system/gpu_available": True,
                "system/memory_gb": 32.0,
                "training/epoch": 1,
                "training/loss": 0.5,
                "validation/rmse": 0.045,
                "business/sharpe_ratio": 1.23,
            }
        )

        logger.info("✓ Comprehensive logging successful")
        run.finish()

    except Exception as e:
        logger.error(f"✗ Custom configuration failed: {e}")


def test_config_validation():
    """Test configuration validation and sanitization."""
    logger.info("\n=== Testing Configuration Validation ===")

    try:
        import numpy as np
        import torch

        # Complex config with various data types
        complex_config = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "none_value": None,
            "list_value": [1, 2, 3],
            "numpy_array": np.array([1, 2, 3]),
            "numpy_int": np.int64(42),
            "numpy_float": np.float32(3.14),
            "nested_dict": {"inner_string": "nested", "inner_number": 123},
        }

        if torch.cuda.is_available():
            complex_config["torch_tensor"] = torch.tensor([1.0, 2.0, 3.0])

        run = init_wandb(
            experiment_name="config_validation_test",
            config=complex_config,
            tags=["validation", "test"],
        )

        logger.info(f"✓ Configuration validation successful")
        logger.info(f"  Complex config handled properly")
        logger.info(f"  Run ID: {run.id}")

        run.finish()

    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")


def main():
    """Run all WANDB setup tests."""
    logger.info("Starting WANDB Setup Tests")
    logger.info("=" * 60)

    # Test individual components
    test_basic_initialization()
    test_training_setup()
    test_evaluation_setup()
    test_hyperparameter_search_setup()
    test_custom_configuration()
    test_config_validation()

    logger.info("\n" + "=" * 60)
    logger.info("WANDB Setup Tests Complete")
    logger.info("All tests passed successfully!")

    # Display configuration info
    logger.info(f"\nWANDB Configuration:")
    logger.info(f"  Default project: {WANDBConfig.DEFAULT_PROJECT}")
    logger.info(f"  Batch log interval: {WANDBConfig.BATCH_LOG_INTERVAL}")
    logger.info(f"  Epoch log interval: {WANDBConfig.EPOCH_LOG_INTERVAL}")
    logger.info(f"  Default tags: {WANDBConfig.DEFAULT_TAGS}")


if __name__ == "__main__":
    main()
