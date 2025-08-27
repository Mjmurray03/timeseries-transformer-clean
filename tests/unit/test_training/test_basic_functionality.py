"""Basic functionality tests for training components."""

import pytest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestTrainingConfigIntegration:
    """Test training configuration integration."""
    
    def test_training_config_import(self):
        """Test that training config can be imported and used."""
        from config.training_config import TrainingConfig, create_quick_test_config
        
        # Test default config creation
        config = TrainingConfig()
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.use_amp is True
        
        # Test quick test config
        quick_config = create_quick_test_config()
        assert quick_config.num_epochs == 5
        assert quick_config.batch_size == 16
        assert quick_config.experiment_name == "quick_test"
    
    def test_training_config_validation(self):
        """Test training config validation."""
        from config.training_config import TrainingConfig
        
        # Test invalid epochs
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=0)
        
        # Test invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)
        
        # Test invalid splits
        with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
            TrainingConfig(train_split=0.5, val_split=0.3, test_split=0.3)
    
    def test_training_config_serialization(self):
        """Test training config serialization."""
        from config.training_config import TrainingConfig
        
        config = TrainingConfig(num_epochs=50, batch_size=64)
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['num_epochs'] == 50
        assert config_dict['batch_size'] == 64
        assert 'optimizer' in config_dict
        assert 'scheduler' in config_dict
        assert 'loss' in config_dict
        
        # Test from_dict
        new_config = TrainingConfig.from_dict(config_dict)
        assert new_config.num_epochs == 50
        assert new_config.batch_size == 64


class TestCallbackLogic:
    """Test callback logic without torch dependencies."""
    
    def test_early_stopping_logic(self):
        """Test early stopping logic."""
        # Simple early stopping implementation for testing
        class SimpleEarlyStopping:
            def __init__(self, patience=3, min_delta=0.0, mode='min'):
                self.patience = patience
                self.min_delta = min_delta
                self.mode = mode
                self.wait = 0
                self.best_score = None
                
                if mode == 'min':
                    self.monitor_op = lambda current, best: current < (best - min_delta)
                elif mode == 'max':
                    self.monitor_op = lambda current, best: current > (best + min_delta)
                else:
                    raise ValueError("Mode must be 'min' or 'max'")
            
            def should_stop(self, current_score):
                if self.best_score is None:
                    self.best_score = current_score
                    return False
                
                if self.monitor_op(current_score, self.best_score):
                    self.best_score = current_score
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        return True
                return False
        
        # Test min mode
        early_stopping = SimpleEarlyStopping(patience=2, mode='min')
        
        assert not early_stopping.should_stop(1.0)  # First score
        assert not early_stopping.should_stop(0.8)  # Improvement
        assert not early_stopping.should_stop(0.9)  # No improvement, wait=1
        assert early_stopping.should_stop(1.0)      # No improvement, wait=2, should stop
        
        # Test max mode
        early_stopping_max = SimpleEarlyStopping(patience=2, mode='max')
        
        assert not early_stopping_max.should_stop(0.5)  # First score
        assert not early_stopping_max.should_stop(0.7)  # Improvement
        assert not early_stopping_max.should_stop(0.6)  # No improvement, wait=1
        assert early_stopping_max.should_stop(0.5)      # No improvement, wait=2, should stop
    
    def test_checkpoint_logic(self):
        """Test checkpoint saving logic."""
        class SimpleCheckpoint:
            def __init__(self, mode='min'):
                self.mode = mode
                self.best_score = None
                
                if mode == 'min':
                    self.monitor_op = lambda current, best: current < best
                elif mode == 'max':
                    self.monitor_op = lambda current, best: current > best
                else:
                    raise ValueError("Mode must be 'min' or 'max'")
            
            def should_save(self, current_score):
                if self.best_score is None:
                    self.best_score = current_score
                    return True
                
                if self.monitor_op(current_score, self.best_score):
                    self.best_score = current_score
                    return True
                
                return False
        
        # Test min mode
        checkpoint = SimpleCheckpoint(mode='min')
        
        assert checkpoint.should_save(1.0)   # First score, should save
        assert checkpoint.should_save(0.8)   # Improvement, should save
        assert not checkpoint.should_save(0.9)  # No improvement, should not save
        assert checkpoint.should_save(0.7)   # Improvement, should save
        
        # Test max mode
        checkpoint_max = SimpleCheckpoint(mode='max')
        
        assert checkpoint_max.should_save(0.5)   # First score, should save
        assert checkpoint_max.should_save(0.7)   # Improvement, should save
        assert not checkpoint_max.should_save(0.6)  # No improvement, should not save
        assert checkpoint_max.should_save(0.8)   # Improvement, should save


class TestDatasetLogic:
    """Test dataset logic without torch dependencies."""
    
    def test_sequence_splitting(self):
        """Test sequence splitting logic."""
        import numpy as np
        
        def split_sequences_simple(sequences, targets, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
            """Simple sequence splitting for testing."""
            assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
            
            n_samples = len(sequences)
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            return (
                (sequences[:train_end], targets[:train_end]),
                (sequences[train_end:val_end], targets[train_end:val_end]),
                (sequences[val_end:], targets[val_end:])
            )
        
        # Test splitting
        sequences = np.arange(100).reshape(100, 1)
        targets = np.arange(100)
        
        (train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets) = split_sequences_simple(
            sequences, targets
        )
        
        assert len(train_seq) == 70
        assert len(val_seq) == 15
        assert len(test_seq) == 15
        assert len(train_targets) == 70
        assert len(val_targets) == 15
        assert len(test_targets) == 15
        
        # Test invalid ratios
        with pytest.raises(AssertionError):
            split_sequences_simple(sequences, targets, 0.5, 0.3, 0.3)
    
    def test_data_augmentation_logic(self):
        """Test data augmentation probability logic."""
        import random
        
        class SimpleAugmentation:
            def __init__(self, noise_prob=0.5, dropout_prob=0.3):
                self.noise_prob = noise_prob
                self.dropout_prob = dropout_prob
            
            def should_apply_noise(self):
                return random.random() < self.noise_prob
            
            def should_apply_dropout(self):
                return random.random() < self.dropout_prob
        
        # Test with high probabilities
        aug_high = SimpleAugmentation(noise_prob=1.0, dropout_prob=1.0)
        assert aug_high.should_apply_noise()
        assert aug_high.should_apply_dropout()
        
        # Test with zero probabilities
        aug_zero = SimpleAugmentation(noise_prob=0.0, dropout_prob=0.0)
        assert not aug_zero.should_apply_noise()
        assert not aug_zero.should_apply_dropout()


class TestMetricsCalculation:
    """Test metrics calculation logic."""
    
    def test_basic_metrics(self):
        """Test basic regression metrics calculation."""
        import numpy as np
        
        def calculate_metrics(predictions, targets):
            """Simple metrics calculation."""
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            return {'mse': mse, 'rmse': rmse, 'mae': mae}
        
        # Perfect predictions
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = calculate_metrics(predictions, targets)
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        
        # Non-perfect predictions
        predictions = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = calculate_metrics(predictions, targets)
        assert abs(metrics['mse'] - 0.01) < 1e-10  # (0.1)^2
        assert abs(metrics['rmse'] - 0.1) < 1e-10
        assert abs(metrics['mae'] - 0.1) < 1e-10
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        import numpy as np
        
        def calculate_directional_accuracy(predictions, targets):
            """Calculate directional accuracy for multi-step predictions."""
            if predictions.shape[1] <= 1:
                return 0.0
            
            pred_direction = np.diff(predictions, axis=1) > 0
            target_direction = np.diff(targets, axis=1) > 0
            
            return np.mean(pred_direction == target_direction)
        
        # Perfect directional predictions
        predictions = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])  # Up, Up | Down, Up
        targets = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])      # Up, Up | Down, Up
        
        accuracy = calculate_directional_accuracy(predictions, targets)
        assert accuracy == 1.0
        
        # Opposite directions
        predictions = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])  # Up, Up | Down, Up
        targets = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 1.0]])      # Down, Down | Up, Down
        
        accuracy = calculate_directional_accuracy(predictions, targets)
        assert accuracy == 0.0


class TestFileOperations:
    """Test file operations and utilities."""
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        import yaml
        
        config_data = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': {
                'name': 'adamw',
                'weight_decay': 0.01
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            # Save config
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load config
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == config_data
    
    def test_metrics_logging(self):
        """Test metrics logging to CSV."""
        from datetime import datetime
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / 'metrics.csv'
            
            # Initialize log file
            with open(log_file, 'w') as f:
                f.write("timestamp,step,metric,value\n")
            
            # Log some metrics
            metrics = {'loss': 0.5, 'accuracy': 0.8}
            timestamp = datetime.now().isoformat()
            
            with open(log_file, 'a') as f:
                for metric, value in metrics.items():
                    f.write(f"{timestamp},1,{metric},{value}\n")
            
            # Read and verify
            content = log_file.read_text()
            lines = content.strip().split('\n')
            
            assert len(lines) == 3  # Header + 2 metrics
            assert 'loss,0.5' in content
            assert 'accuracy,0.8' in content