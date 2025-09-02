"""Unit tests for training callbacks."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Mock torch before importing
import sys
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.optim'] = Mock()
sys.modules['torch.optim.lr_scheduler'] = Mock()
sys.modules['torch.cuda'] = Mock()
sys.modules['torch.cuda.amp'] = Mock()

from src.training.callbacks.early_stopping import EarlyStopping
from src.training.callbacks.model_checkpoint import ModelCheckpoint


class TestEarlyStopping:
    """Test suite for EarlyStopping callback."""
    
    def test_initialization(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=5, min_delta=0.01, mode='min')
        
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 0.01
        assert early_stopping.mode == 'min'
        assert early_stopping.wait == 0
        assert early_stopping.best_score is None
    
    def test_initialization_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="Mode must be 'min' or 'max'"):
            EarlyStopping(mode='invalid')
    
    def test_should_stop_min_mode_improvement(self):
        """Test early stopping with min mode and improvement."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # First score - should not stop
        assert not early_stopping.should_stop(1.0)
        assert early_stopping.best_score == 1.0
        assert early_stopping.wait == 0
        
        # Improvement - should not stop
        assert not early_stopping.should_stop(0.8)
        assert early_stopping.best_score == 0.8
        assert early_stopping.wait == 0
        
        # Another improvement - should not stop
        assert not early_stopping.should_stop(0.6)
        assert early_stopping.best_score == 0.6
        assert early_stopping.wait == 0
    
    def test_should_stop_min_mode_no_improvement(self):
        """Test early stopping with min mode and no improvement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='min')
        
        # Set initial best score
        early_stopping.should_stop(1.0)
        
        # No improvement - wait should increase
        assert not early_stopping.should_stop(1.1)
        assert early_stopping.wait == 1
        
        # Still no improvement - should trigger stopping
        assert early_stopping.should_stop(1.2)
        assert early_stopping.wait == 2
    
    def test_should_stop_max_mode(self):
        """Test early stopping with max mode."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        
        # First score
        assert not early_stopping.should_stop(0.5)
        
        # Improvement (higher is better)
        assert not early_stopping.should_stop(0.7)
        assert early_stopping.best_score == 0.7
        assert early_stopping.wait == 0
        
        # No improvement
        assert not early_stopping.should_stop(0.6)
        assert early_stopping.wait == 1
        
        # Still no improvement - should stop
        assert early_stopping.should_stop(0.5)
    
    def test_min_delta_threshold(self):
        """Test min_delta threshold behavior."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, mode='min')
        
        # Set initial score
        early_stopping.should_stop(1.0)
        
        # Small improvement (less than min_delta) - should count as no improvement
        assert not early_stopping.should_stop(0.95)
        assert early_stopping.wait == 1
        
        # Significant improvement (greater than min_delta) - should reset wait
        assert not early_stopping.should_stop(0.8)
        assert early_stopping.wait == 0
    
    def test_reset(self):
        """Test resetting early stopping state."""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # Set some state
        early_stopping.should_stop(1.0)
        early_stopping.should_stop(1.1)
        
        assert early_stopping.wait == 1
        assert early_stopping.best_score == 1.0
        
        # Reset
        early_stopping.reset()
        
        assert early_stopping.wait == 0
        assert early_stopping.best_score is None
    
    def test_get_best_score(self):
        """Test getting best score."""
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        assert early_stopping.get_best_score() is None
        
        early_stopping.should_stop(1.0)
        assert early_stopping.get_best_score() == 1.0
        
        early_stopping.should_stop(0.8)
        assert early_stopping.get_best_score() == 0.8


class TestModelCheckpoint:
    """Test suite for ModelCheckpoint callback."""
    
    def test_initialization(self):
        """Test model checkpoint initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            assert checkpoint.checkpoint_dir == Path(temp_dir)
            assert checkpoint.monitor == 'val_loss'
            assert checkpoint.mode == 'min'
            assert checkpoint.save_best_only is True
            assert checkpoint.best_score is None
    
    def test_initialization_invalid_mode(self):
        """Test initialization with invalid mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Mode must be 'min' or 'max'"):
                ModelCheckpoint(checkpoint_dir=temp_dir, mode='invalid')
    
    def test_on_epoch_end_missing_metric(self):
        """Test epoch end with missing monitored metric."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(checkpoint_dir=temp_dir, monitor='val_loss')
            
            # Mock model and optimizer
            model = Mock()
            model.state_dict.return_value = {'param': 'value'}
            optimizer = Mock()
            optimizer.state_dict.return_value = {'lr': 0.001}
            
            metrics = {'train_loss': 0.5}  # Missing val_loss
            
            # Should handle gracefully
            checkpoint.on_epoch_end(0, model, optimizer, metrics)
            
            # No checkpoint should be saved
            assert len(list(Path(temp_dir).glob('*.pt'))) == 0
    
    @patch('torch.save')
    def test_on_epoch_end_first_epoch(self, mock_torch_save):
        """Test epoch end for first epoch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            # Mock model and optimizer
            model = Mock()
            model.state_dict.return_value = {'param': 'value'}
            optimizer = Mock()
            optimizer.state_dict.return_value = {'lr': 0.001}
            
            metrics = {'val_loss': 0.5}
            
            checkpoint.on_epoch_end(0, model, optimizer, metrics)
            
            # Should save checkpoint (first epoch is always best)
            assert mock_torch_save.called
            assert checkpoint.best_score == 0.5
    
    @patch('torch.save')
    def test_on_epoch_end_improvement(self, mock_torch_save):
        """Test epoch end with improvement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            # Mock model and optimizer
            model = Mock()
            model.state_dict.return_value = {'param': 'value'}
            optimizer = Mock()
            optimizer.state_dict.return_value = {'lr': 0.001}
            
            # First epoch
            checkpoint.on_epoch_end(0, model, optimizer, {'val_loss': 1.0})
            
            # Reset mock to count only second call
            mock_torch_save.reset_mock()
            
            # Second epoch with improvement
            checkpoint.on_epoch_end(1, model, optimizer, {'val_loss': 0.8})
            
            # Should save checkpoint
            assert mock_torch_save.called
            assert checkpoint.best_score == 0.8
    
    @patch('torch.save')
    def test_on_epoch_end_no_improvement(self, mock_torch_save):
        """Test epoch end without improvement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            # Mock model and optimizer
            model = Mock()
            model.state_dict.return_value = {'param': 'value'}
            optimizer = Mock()
            optimizer.state_dict.return_value = {'lr': 0.001}
            
            # First epoch
            checkpoint.on_epoch_end(0, model, optimizer, {'val_loss': 0.5})
            
            # Reset mock
            mock_torch_save.reset_mock()
            
            # Second epoch without improvement
            checkpoint.on_epoch_end(1, model, optimizer, {'val_loss': 0.7})
            
            # Should not save checkpoint (save_best_only=True and no improvement)
            # But should save last checkpoint
            assert checkpoint.best_score == 0.5  # Should remain unchanged
    
    @patch('torch.save')
    def test_save_last_checkpoint(self, mock_torch_save):
        """Test saving last checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                monitor='val_loss',
                save_last=True
            )
            
            # Mock model and optimizer
            model = Mock()
            model.state_dict.return_value = {'param': 'value'}
            optimizer = Mock()
            optimizer.state_dict.return_value = {'lr': 0.001}
            
            checkpoint.on_epoch_end(0, model, optimizer, {'val_loss': 0.5})
            
            # Should save both regular checkpoint and last checkpoint
            assert mock_torch_save.call_count >= 1
    
    def test_get_best_score(self):
        """Test getting best score."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(checkpoint_dir=temp_dir)
            
            assert checkpoint.get_best_score() is None
            
            # Set best score
            checkpoint.best_score = 0.5
            assert checkpoint.get_best_score() == 0.5