"""
Comprehensive unit tests for training pipeline components.
Follows patterns from testing-standards.md with 85% coverage requirement.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.training.trainer import Trainer
from src.training.callbacks.early_stopping import EarlyStopping
from src.training.callbacks.model_checkpoint import ModelCheckpoint
from src.training.experiment_tracker import ExperimentTracker
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.models.losses.composite_loss import CompositeLoss


class TestTrainer:
    """Test suite for Trainer following testing-standards.md patterns"""
    
    @pytest.fixture
    def training_config(self):
        """Create training configuration for testing"""
        return TrainingConfig(
            learning_rate=1e-4,
            batch_size=16,
            num_epochs=5,
            patience=3,
            gradient_clip=1.0,
            weight_decay=1e-5,
            warmup_steps=10,
            save_every=2
        )
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing"""
        return ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            forecast_horizon=5
        )
    
    @pytest.fixture
    def test_model(self, model_config):
        """Create test model instance"""
        return TimeSeriesTransformer(**model_config.__dict__)
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for testing"""
        def generate_batch():
            return {
                'input': torch.randn(16, 60, 7),
                'target': torch.randn(16, 5)
            }
        
        # Create mock dataloader that yields batches
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([generate_batch() for _ in range(10)]))
        mock_loader.__len__ = Mock(return_value=10)
        return mock_loader
    
    @pytest.fixture
    def trainer(self, test_model, training_config, mock_dataloader, tmp_path):
        """Create Trainer instance for testing"""
        return Trainer(
            model=test_model,
            config=training_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            save_dir=str(tmp_path)
        )
    
    def test_initialization(self, trainer, training_config):
        """Test trainer initializes correctly"""
        assert trainer is not None
        assert trainer.config == training_config
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert hasattr(trainer, 'train_loader')
        assert hasattr(trainer, 'val_loader')
    
    def test_happy_path(self, trainer):
        """Test normal training operation succeeds"""
        # Mock experiment tracking to avoid external dependencies
        with patch.object(trainer, 'experiment_tracker') as mock_tracker:
            mock_tracker.log_metrics = Mock()
            mock_tracker.log_model = Mock()
            
            history = trainer.fit()
            
            # Check training completed
            assert isinstance(history, dict)
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) > 0
            assert len(history['val_loss']) > 0
            
            # Check metrics are reasonable
            assert all(loss > 0 for loss in history['train_loss'])
            assert all(loss > 0 for loss in history['val_loss'])
    
    def test_single_epoch_training(self, trainer):
        """Test single epoch training step"""
        trainer.config.num_epochs = 1
        
        with patch.object(trainer, 'experiment_tracker'):
            history = trainer.fit()
            
            assert len(history['train_loss']) == 1
            assert len(history['val_loss']) == 1
    
    def test_edge_cases(self, test_model, training_config, tmp_path):
        """Test boundary conditions"""
        # Empty dataloader
        empty_loader = Mock()
        empty_loader.__iter__ = Mock(return_value=iter([]))
        empty_loader.__len__ = Mock(return_value=0)
        
        trainer = Trainer(
            model=test_model,
            config=training_config,
            train_loader=empty_loader,
            val_loader=empty_loader,
            save_dir=str(tmp_path)
        )
        
        # Should handle empty dataloader gracefully
        with patch.object(trainer, 'experiment_tracker'):
            try:
                history = trainer.fit()
                # If it doesn't raise an exception, history should be empty or minimal
                assert isinstance(history, dict)
            except (StopIteration, RuntimeError):
                # Expected behavior for empty dataloader
                pass
        
        # Single batch training
        single_batch_loader = Mock()
        batch = {
            'input': torch.randn(16, 60, 7),
            'target': torch.randn(16, 5)
        }
        single_batch_loader.__iter__ = Mock(return_value=iter([batch]))
        single_batch_loader.__len__ = Mock(return_value=1)
        
        single_trainer = Trainer(
            model=test_model,
            config=training_config,
            train_loader=single_batch_loader,
            val_loader=single_batch_loader,
            save_dir=str(tmp_path)
        )
        
        with patch.object(single_trainer, 'experiment_tracker'):
            history = single_trainer.fit()
            assert isinstance(history, dict)
    
    def test_error_handling(self, trainer):
        """Test error conditions raise appropriately"""
        # Test with invalid model state
        trainer.model = None
        
        with pytest.raises(AttributeError):
            trainer.fit()
        
        # Test with corrupted batch data
        def bad_batch_generator():
            yield {'input': torch.randn(16, 60, 7)}  # Missing target
        
        bad_loader = Mock()
        bad_loader.__iter__ = Mock(return_value=bad_batch_generator())
        bad_loader.__len__ = Mock(return_value=1)
        
        # Reset model
        from src.models.timeseries_transformer import TimeSeriesTransformer
        from src.config.model_config import ModelConfig
        
        model_config = ModelConfig(
            sequence_length=60, num_features=7, d_model=128,
            n_heads=4, n_layers=2, dropout=0.1, forecast_horizon=5
        )
        trainer.model = TimeSeriesTransformer(**model_config.__dict__)
        trainer.train_loader = bad_loader
        
        with patch.object(trainer, 'experiment_tracker'):
            with pytest.raises(KeyError):
                trainer.fit()
    
    def test_gradient_clipping(self, trainer):
        """Test gradient clipping is applied correctly"""
        original_clip_value = trainer.config.gradient_clip
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            mock_clip.return_value = torch.tensor(2.0)  # Simulate clipped norm
            
            with patch.object(trainer, 'experiment_tracker'):
                trainer._train_epoch(epoch=1)
                
                # Verify gradient clipping was called
                assert mock_clip.called
                # Check it was called with correct parameters
                call_args = mock_clip.call_args
                assert call_args[1]['max_norm'] == original_clip_value
    
    def test_learning_rate_scheduling(self, trainer):
        """Test learning rate scheduling works correctly"""
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        with patch.object(trainer, 'experiment_tracker'):
            # Train for a few steps to trigger scheduler
            trainer._train_epoch(epoch=1)
            trainer.scheduler.step()
            
            # Learning rate should have changed (depends on scheduler type)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            # Note: The exact change depends on the scheduler implementation
            assert isinstance(current_lr, float)
            assert current_lr > 0
    
    def test_loss_computation(self, trainer):
        """Test loss computation is working correctly"""
        # Get a batch from dataloader
        batch = next(iter(trainer.train_loader))
        
        trainer.model.train()
        
        # Forward pass
        outputs = trainer.model(batch['input'])
        loss = trainer.criterion(outputs, batch['target'])
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Positive loss
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
    
    def test_model_saving_loading(self, trainer):
        """Test model checkpointing functionality"""
        checkpoint_path = Path(trainer.save_dir) / "test_checkpoint.pt"
        
        # Save model
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, loss=1.5)
        assert checkpoint_path.exists()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify checkpoint contents
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint
        
        assert checkpoint['epoch'] == 1
        assert checkpoint['loss'] == 1.5


class TestEarlyStopping:
    """Test suite for EarlyStopping callback"""
    
    @pytest.fixture
    def early_stopping(self):
        """Create EarlyStopping instance for testing"""
        return EarlyStopping(patience=3, min_delta=1e-4, restore_best_weights=True)
    
    def test_initialization(self, early_stopping):
        """Test early stopping initializes correctly"""
        assert early_stopping is not None
        assert early_stopping.patience == 3
        assert early_stopping.min_delta == 1e-4
        assert early_stopping.restore_best_weights is True
        assert early_stopping.best_loss == float('inf')
        assert early_stopping.wait == 0
        assert early_stopping.stopped_epoch == 0
    
    def test_happy_path(self, early_stopping):
        """Test normal early stopping behavior"""
        # Simulate improving validation loss
        losses = [1.0, 0.9, 0.85, 0.8, 0.78]
        
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss, epoch=i+1)
            assert should_stop is False  # Should not stop while improving
            assert early_stopping.best_loss <= loss + early_stopping.min_delta
    
    def test_edge_cases(self, early_stopping):
        """Test boundary conditions"""
        # No improvement case
        losses = [1.0, 1.1, 1.05, 1.08, 1.12]  # Consistently bad or getting worse
        
        should_stop = False
        for i, loss in enumerate(losses):
            should_stop = early_stopping(loss, epoch=i+1)
            if should_stop:
                break
        
        assert should_stop is True  # Should stop due to no improvement
        assert early_stopping.wait >= early_stopping.patience
    
    def test_improvement_detection(self, early_stopping):
        """Test improvement detection with min_delta"""
        # Test marginal improvement (less than min_delta)
        marginal_losses = [1.0, 0.99999, 0.99998]  # Very small improvements
        
        for i, loss in enumerate(marginal_losses[:-1]):
            early_stopping(loss, epoch=i+1)
        
        # This should not be considered an improvement
        last_wait = early_stopping.wait
        early_stopping(marginal_losses[-1], epoch=len(marginal_losses))
        
        # Wait count should increase (no significant improvement)
        assert early_stopping.wait > last_wait
    
    def test_reset_on_improvement(self, early_stopping):
        """Test wait counter resets on significant improvement"""
        # First, build up some wait count
        bad_losses = [1.0, 1.1, 1.05]
        for i, loss in enumerate(bad_losses):
            early_stopping(loss, epoch=i+1)
        
        assert early_stopping.wait > 0
        
        # Then show significant improvement
        early_stopping(0.5, epoch=4)  # Significant improvement
        
        assert early_stopping.wait == 0  # Should reset
        assert early_stopping.best_loss == 0.5
    
    def test_stopped_epoch_tracking(self, early_stopping):
        """Test tracking of when stopping occurred"""
        losses = [1.0, 1.1, 1.1, 1.1, 1.1]  # No improvement
        
        stopped = False
        for i, loss in enumerate(losses):
            stopped = early_stopping(loss, epoch=i+1)
            if stopped:
                break
        
        assert stopped is True
        assert early_stopping.stopped_epoch > 0


class TestModelCheckpoint:
    """Test suite for ModelCheckpoint callback"""
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary directory for checkpoints"""
        return tmp_path
    
    @pytest.fixture
    def model_checkpoint(self, checkpoint_dir):
        """Create ModelCheckpoint instance for testing"""
        return ModelCheckpoint(
            filepath=str(checkpoint_dir / "checkpoint_{epoch:02d}_{val_loss:.4f}.pt"),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    
    @pytest.fixture
    def test_model(self):
        """Create simple test model"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def test_initialization(self, model_checkpoint):
        """Test checkpoint callback initializes correctly"""
        assert model_checkpoint is not None
        assert model_checkpoint.monitor == 'val_loss'
        assert model_checkpoint.save_best_only is True
        assert model_checkpoint.mode == 'min'
        assert model_checkpoint.best == float('inf')
    
    def test_happy_path(self, model_checkpoint, test_model, checkpoint_dir):
        """Test normal checkpointing behavior"""
        # Simulate training with improving validation loss
        losses = [1.0, 0.9, 0.8, 0.75, 0.7]
        
        for epoch, val_loss in enumerate(losses, 1):
            logs = {'val_loss': val_loss}
            model_checkpoint.on_epoch_end(epoch, test_model, logs)
            
            # Check if checkpoint was saved (should save when improving)
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
            
            if val_loss < model_checkpoint.best:
                assert len(checkpoint_files) > 0
    
    def test_best_only_behavior(self, test_model, checkpoint_dir):
        """Test save_best_only functionality"""
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.pt"),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # First epoch - should save as it's the first
        checkpoint.on_epoch_end(1, test_model, {'val_loss': 1.0})
        assert (checkpoint_dir / "best_model.pt").exists()
        
        # Second epoch - worse performance, should not save
        checkpoint.on_epoch_end(2, test_model, {'val_loss': 1.5})
        # File should still be from epoch 1
        
        # Third epoch - better performance, should save
        initial_mtime = (checkpoint_dir / "best_model.pt").stat().st_mtime
        import time
        time.sleep(0.1)  # Ensure different timestamp
        checkpoint.on_epoch_end(3, test_model, {'val_loss': 0.8})
        new_mtime = (checkpoint_dir / "best_model.pt").stat().st_mtime
        
        assert new_mtime > initial_mtime  # File was updated
    
    def test_save_all_behavior(self, test_model, checkpoint_dir):
        """Test save_best_only=False functionality"""
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_dir / "model_{epoch:02d}.pt"),
            save_best_only=False
        )
        
        # Should save every epoch regardless of performance
        for epoch in range(1, 4):
            checkpoint.on_epoch_end(epoch, test_model, {'val_loss': 1.0})
            expected_file = checkpoint_dir / f"model_{epoch:02d}.pt"
            assert expected_file.exists()
    
    def test_mode_max_behavior(self, test_model, checkpoint_dir):
        """Test mode='max' for metrics where higher is better"""
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_accuracy.pt"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Should save when accuracy improves (increases)
        accuracies = [0.7, 0.8, 0.75, 0.85, 0.9]
        
        for epoch, acc in enumerate(accuracies, 1):
            checkpoint.on_epoch_end(epoch, test_model, {'val_accuracy': acc})
        
        # Should have saved the best accuracy (0.9)
        assert checkpoint.best == 0.9
    
    def test_file_format_strings(self, test_model, checkpoint_dir):
        """Test filename formatting with variables"""
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_dir / "model_e{epoch:03d}_loss{val_loss:.6f}.pt"),
            save_best_only=False
        )
        
        checkpoint.on_epoch_end(5, test_model, {'val_loss': 0.123456})
        
        # Check that file was created with correct formatting
        expected_files = list(checkpoint_dir.glob("model_e005_loss*.pt"))
        assert len(expected_files) == 1
        
        filename = expected_files[0].name
        assert "model_e005_loss0.123456.pt" == filename


class TestExperimentTracker:
    """Test suite for ExperimentTracker"""
    
    @pytest.fixture
    def experiment_config(self):
        """Create experiment configuration"""
        return {
            'project_name': 'test_timeseries',
            'experiment_name': 'test_run',
            'tags': ['test', 'unit_test'],
            'log_frequency': 10
        }
    
    @pytest.fixture
    def experiment_tracker(self, experiment_config):
        """Create ExperimentTracker instance for testing"""
        # Mock W&B to avoid external dependencies
        with patch('wandb.init'), patch('wandb.log'), patch('wandb.finish'):
            return ExperimentTracker(**experiment_config)
    
    def test_initialization(self, experiment_tracker, experiment_config):
        """Test experiment tracker initializes correctly"""
        assert experiment_tracker is not None
        assert experiment_tracker.project_name == experiment_config['project_name']
        assert experiment_tracker.experiment_name == experiment_config['experiment_name']
        assert experiment_tracker.tags == experiment_config['tags']
    
    @patch('wandb.log')
    def test_log_metrics(self, mock_wandb_log, experiment_tracker):
        """Test metrics logging functionality"""
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6,
            'learning_rate': 1e-4,
            'epoch': 1
        }
        
        experiment_tracker.log_metrics(metrics, step=1)
        
        # Verify W&B log was called
        mock_wandb_log.assert_called_once()
        call_args = mock_wandb_log.call_args
        assert 'step' in call_args[1]
        assert call_args[1]['step'] == 1
    
    @patch('wandb.log')
    def test_log_model_artifacts(self, mock_wandb_log, experiment_tracker, tmp_path):
        """Test model artifact logging"""
        # Create a dummy model file
        model_path = tmp_path / "model.pt"
        torch.save({'model': 'dummy'}, model_path)
        
        experiment_tracker.log_model_artifact(str(model_path), 'best_model')
        
        # Should have attempted to log something
        # Specific behavior depends on implementation
        assert hasattr(experiment_tracker, 'log_model_artifact')
    
    def test_log_hyperparameters(self, experiment_tracker):
        """Test hyperparameter logging"""
        hyperparams = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_epochs': 100,
            'model_dim': 256
        }
        
        # Should not raise any errors
        experiment_tracker.log_hyperparameters(hyperparams)
    
    def test_log_frequency_throttling(self, experiment_tracker):
        """Test log frequency throttling"""
        # Set a high frequency for testing
        experiment_tracker.log_frequency = 5
        
        # Log multiple times rapidly
        for i in range(10):
            should_log = experiment_tracker._should_log_this_step(i)
            if i % 5 == 0:
                assert should_log is True
            else:
                assert should_log is False
    
    @patch('wandb.finish')
    def test_experiment_completion(self, mock_wandb_finish, experiment_tracker):
        """Test experiment cleanup"""
        experiment_tracker.finish_experiment()
        mock_wandb_finish.assert_called_once()
    
    def test_context_manager(self, experiment_config):
        """Test experiment tracker as context manager"""
        with patch('wandb.init'), patch('wandb.finish') as mock_finish:
            with ExperimentTracker(**experiment_config) as tracker:
                assert tracker is not None
                tracker.log_metrics({'loss': 1.0}, step=1)
            
            # Should automatically finish when exiting context
            mock_finish.assert_called_once()


class TestCompositeLoss:
    """Test suite for CompositeLoss used in training"""
    
    @pytest.fixture
    def composite_loss(self):
        """Create CompositeLoss instance for testing"""
        return CompositeLoss(
            price_weight=0.6,
            direction_weight=0.3,
            quantile_weight=0.1,
            alpha_quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
    
    @pytest.fixture
    def predictions(self):
        """Generate sample predictions"""
        torch.manual_seed(42)
        return {
            'prices': torch.randn(16, 5),  # batch_size, forecast_horizon
            'quantiles': torch.randn(16, 5, 5),  # batch_size, forecast_horizon, num_quantiles
            'direction': torch.sigmoid(torch.randn(16, 5))  # probabilities
        }
    
    @pytest.fixture
    def targets(self):
        """Generate sample targets"""
        torch.manual_seed(42)
        return {
            'prices': torch.randn(16, 5),
            'directions': torch.randint(0, 2, (16, 5)).float()  # Binary directions
        }
    
    def test_initialization(self, composite_loss):
        """Test composite loss initializes correctly"""
        assert composite_loss is not None
        assert composite_loss.price_weight == 0.6
        assert composite_loss.direction_weight == 0.3
        assert composite_loss.quantile_weight == 0.1
        assert len(composite_loss.alpha_quantiles) == 5
    
    def test_happy_path(self, composite_loss, predictions, targets):
        """Test normal loss computation succeeds"""
        loss = composite_loss(predictions, targets)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Positive loss
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
    
    def test_loss_components(self, composite_loss, predictions, targets):
        """Test individual loss components are computed correctly"""
        # Get detailed loss breakdown if available
        if hasattr(composite_loss, 'compute_detailed_loss'):
            detailed_loss = composite_loss.compute_detailed_loss(predictions, targets)
            
            assert 'price_loss' in detailed_loss
            assert 'direction_loss' in detailed_loss
            assert 'quantile_loss' in detailed_loss
            assert 'total_loss' in detailed_loss
            
            # All components should be positive
            for component, value in detailed_loss.items():
                assert value >= 0, f"{component} should be non-negative"
    
    def test_weight_influence(self):
        """Test loss component weights affect total loss correctly"""
        # Create two loss instances with different weights
        high_price_weight = CompositeLoss(price_weight=0.9, direction_weight=0.05, quantile_weight=0.05)
        high_direction_weight = CompositeLoss(price_weight=0.05, direction_weight=0.9, quantile_weight=0.05)
        
        torch.manual_seed(42)
        predictions = {
            'prices': torch.randn(8, 5),
            'quantiles': torch.randn(8, 5, 5),
            'direction': torch.sigmoid(torch.randn(8, 5))
        }
        targets = {
            'prices': torch.randn(8, 5),
            'directions': torch.randint(0, 2, (8, 5)).float()
        }
        
        loss1 = high_price_weight(predictions, targets)
        loss2 = high_direction_weight(predictions, targets)
        
        # Losses should be different due to different weighting
        assert not torch.allclose(loss1, loss2, atol=1e-6)
    
    def test_edge_cases(self, composite_loss):
        """Test boundary conditions"""
        # Perfect predictions (should give low loss)
        torch.manual_seed(42)
        perfect_targets = {
            'prices': torch.randn(4, 5),
            'directions': torch.randint(0, 2, (4, 5)).float()
        }
        perfect_predictions = {
            'prices': perfect_targets['prices'].clone(),
            'quantiles': perfect_targets['prices'].unsqueeze(-1).repeat(1, 1, 5),
            'direction': perfect_targets['directions'].clone()
        }
        
        perfect_loss = composite_loss(perfect_predictions, perfect_targets)
        assert perfect_loss.item() < 1e-2  # Should be very small
        
        # Single sample
        single_targets = {
            'prices': torch.randn(1, 5),
            'directions': torch.randint(0, 2, (1, 5)).float()
        }
        single_predictions = {
            'prices': torch.randn(1, 5),
            'quantiles': torch.randn(1, 5, 5),
            'direction': torch.sigmoid(torch.randn(1, 5))
        }
        
        single_loss = composite_loss(single_predictions, single_targets)
        assert isinstance(single_loss, torch.Tensor)
        assert single_loss.dim() == 0
    
    def test_gradient_computation(self, composite_loss, predictions, targets):
        """Test gradients can be computed through composite loss"""
        # Make predictions require gradients
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.requires_grad_(True)
        
        loss = composite_loss(predictions, targets)
        loss.backward()
        
        # Check gradients exist for all prediction tensors
        for key, tensor in predictions.items():
            if tensor.requires_grad:
                assert tensor.grad is not None, f"No gradient for {key}"
                assert not torch.isnan(tensor.grad).any(), f"NaN gradient for {key}"