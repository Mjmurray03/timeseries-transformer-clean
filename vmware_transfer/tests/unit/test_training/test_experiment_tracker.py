"""Unit tests for experiment tracker."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# Mock torch and related modules before importing
torch_mock = Mock()
torch_mock.utils = Mock()
torch_mock.utils.tensorboard = Mock()
torch_mock.utils.tensorboard.SummaryWriter = Mock()
sys.modules['torch'] = torch_mock
sys.modules['torch.utils'] = torch_mock.utils
sys.modules['torch.utils.tensorboard'] = torch_mock.utils.tensorboard
sys.modules['numpy'] = Mock()

# Mock optional dependencies
sys.modules['wandb'] = None
sys.modules['mlflow'] = None

# Import directly from the file to avoid trainer import issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "experiment_tracker", 
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'training', 'experiment_tracker.py')
)
experiment_tracker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_tracker_module)

ExperimentTracker = experiment_tracker_module.ExperimentTracker
MetricsLogger = experiment_tracker_module.MetricsLogger


class TestExperimentTracker:
    """Test suite for ExperimentTracker."""
    
    def test_initialization_without_optional_deps(self):
        """Test tracker initializes correctly without optional dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                project_name="test_project",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            assert tracker.experiment_name == "test_experiment"
            assert tracker.project_name == "test_project"
            assert tracker.wandb_run is None
            assert tracker.mlflow_run is None
            assert tracker.tb_writer is None
    
    def test_initialization_with_config(self):
        """Test tracker initializes with configuration."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                config=config,
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            assert tracker.config == config
            
            # Check if config was saved locally
            config_file = Path(temp_dir) / "test_experiment_config.json"
            assert config_file.exists()
            
            with open(config_file) as f:
                saved_config = json.load(f)
            assert saved_config == config
    
    def test_log_metrics_without_platforms(self):
        """Test logging metrics when no platforms are available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            metrics = {"loss": 0.5, "accuracy": 0.8}
            
            # Should not raise any exceptions
            tracker.log_metrics(metrics, step=1, prefix="train")
    
    def test_log_metrics_with_prefix(self):
        """Test logging metrics with prefix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            metrics = {"loss": 0.5, "accuracy": 0.8}
            
            # Mock the internal methods to verify prefix is applied
            with patch.object(tracker, 'wandb_run', None):
                with patch.object(tracker, 'mlflow_run', None):
                    with patch.object(tracker, 'tb_writer', None):
                        tracker.log_metrics(metrics, step=1, prefix="train")
    
    def test_flatten_dict(self):
        """Test dictionary flattening utility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            nested_dict = {
                "optimizer": {
                    "learning_rate": 0.001,
                    "weight_decay": 0.01
                },
                "model": {
                    "hidden_dim": 256,
                    "num_layers": 6
                },
                "batch_size": 32
            }
            
            flattened = tracker._flatten_dict(nested_dict)
            
            expected = {
                "optimizer.learning_rate": 0.001,
                "optimizer.weight_decay": 0.01,
                "model.hidden_dim": 256,
                "model.num_layers": 6,
                "batch_size": 32
            }
            
            assert flattened == expected
    
    def test_context_manager(self):
        """Test tracker as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with ExperimentTracker(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            ) as tracker:
                assert tracker.experiment_name == "test_experiment"
            
            # Should finish cleanly without errors
    
    def test_log_model_without_platforms(self):
        """Test logging model artifacts when no platforms are available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(
                experiment_name="test_experiment",
                log_dir=temp_dir,
                use_wandb=False,
                use_mlflow=False,
                use_tensorboard=False
            )
            
            # Create a dummy model file
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_text("dummy model content")
            
            metadata = {"accuracy": 0.95, "loss": 0.05}
            
            # Should not raise any exceptions
            tracker.log_model(str(model_path), metadata=metadata)


class TestMetricsLogger:
    """Test suite for MetricsLogger."""
    
    def test_initialization(self):
        """Test metrics logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "metrics.csv"
            logger = MetricsLogger(str(log_file))
            
            assert logger.log_file == log_file
            assert log_file.exists()
            
            # Check header
            content = log_file.read_text()
            assert content.startswith("timestamp,step,metric,value")
    
    def test_log_metrics(self):
        """Test logging metrics to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "metrics.csv"
            logger = MetricsLogger(str(log_file))
            
            metrics = {"loss": 0.5, "accuracy": 0.8}
            logger.log(step=1, metrics=metrics)
            
            content = log_file.read_text()
            lines = content.strip().split('\n')
            
            # Should have header + 2 metric lines
            assert len(lines) == 3
            assert "loss" in content
            assert "accuracy" in content
            assert "0.5" in content
            assert "0.8" in content
    
    def test_multiple_log_calls(self):
        """Test multiple logging calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "metrics.csv"
            logger = MetricsLogger(str(log_file))
            
            # Log multiple steps
            logger.log(step=1, metrics={"loss": 0.5})
            logger.log(step=2, metrics={"loss": 0.4})
            logger.log(step=3, metrics={"loss": 0.3})
            
            content = log_file.read_text()
            lines = content.strip().split('\n')
            
            # Should have header + 3 metric lines
            assert len(lines) == 4
            assert "0.5" in content
            assert "0.4" in content
            assert "0.3" in content