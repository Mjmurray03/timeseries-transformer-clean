"""Unit tests for training configuration."""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.config.training_config import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    TrainingConfigValidator,
    create_default_config,
    create_quick_test_config,
    create_production_config
)


class TestOptimizerConfig:
    """Test optimizer configuration."""
    
    def test_default_values(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        
        assert config.name == "adamw"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad is False
    
    def test_custom_values(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.001,
            betas=(0.8, 0.99),
            eps=1e-7,
            amsgrad=True
        )
        
        assert config.name == "adam"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.001
        assert config.betas == (0.8, 0.99)
        assert config.eps == 1e-7
        assert config.amsgrad is True


class TestSchedulerConfig:
    """Test scheduler configuration."""
    
    def test_default_values(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        
        assert config.name == "cosine"
        assert config.warmup_steps == 1000
        assert config.max_steps == 10000
        assert config.min_lr == 1e-6
        assert config.patience == 10
        assert config.factor == 0.5
    
    def test_custom_values(self):
        """Test custom scheduler configuration."""
        config = SchedulerConfig(
            name="step",
            warmup_steps=500,
            max_steps=5000,
            min_lr=1e-7,
            patience=5,
            factor=0.8
        )
        
        assert config.name == "step"
        assert config.warmup_steps == 500
        assert config.max_steps == 5000
        assert config.min_lr == 1e-7
        assert config.patience == 5
        assert config.factor == 0.8


class TestLossConfig:
    """Test loss configuration."""
    
    def test_default_values(self):
        """Test default loss configuration."""
        config = LossConfig()
        
        assert config.price_loss_weight == 1.0
        assert config.direction_loss_weight == 0.5
        assert config.volatility_loss_weight == 0.3
        assert config.quantile_loss_weight == 0.2
        assert config.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
    
    def test_custom_values(self):
        """Test custom loss configuration."""
        config = LossConfig(
            price_loss_weight=2.0,
            direction_loss_weight=1.0,
            volatility_loss_weight=0.5,
            quantile_loss_weight=0.3,
            quantiles=[0.25, 0.5, 0.75]
        )
        
        assert config.price_loss_weight == 2.0
        assert config.direction_loss_weight == 1.0
        assert config.volatility_loss_weight == 0.5
        assert config.quantile_loss_weight == 0.3
        assert config.quantiles == [0.25, 0.5, 0.75]


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.gradient_accumulation_steps == 1
        assert config.gradient_clip == 1.0
        assert config.use_amp is True
        assert config.device == "cuda"
        assert config.train_split == 0.7
        assert config.val_split == 0.15
        assert config.test_split == 0.15
        assert config.seed == 42
        assert config.deterministic is True
    
    def test_validation_success(self):
        """Test successful validation."""
        config = TrainingConfig(
            num_epochs=50,
            batch_size=16,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_validation_splits_sum_error(self):
        """Test validation error for splits not summing to 1.0."""
        with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
            TrainingConfig(
                train_split=0.8,
                val_split=0.1,
                test_split=0.2  # Sum = 1.1
            )
    
    def test_validation_negative_epochs(self):
        """Test validation error for negative epochs."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TrainingConfig(num_epochs=-1)
    
    def test_validation_negative_batch_size(self):
        """Test validation error for negative batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)
    
    def test_validation_negative_learning_rate(self):
        """Test validation error for negative learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(
                optimizer=OptimizerConfig(learning_rate=-1e-4)
            )
    
    def test_validation_invalid_quantiles(self):
        """Test validation error for invalid quantiles."""
        with pytest.raises(ValueError, match="All quantiles must be between 0 and 1"):
            TrainingConfig(
                loss=LossConfig(quantiles=[0.1, 0.5, 1.5])
            )
    
    def test_validation_invalid_device(self):
        """Test validation error for invalid device."""
        with pytest.raises(ValueError, match="Unsupported device"):
            TrainingConfig(device="invalid")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['num_epochs'] == 100
        assert config_dict['batch_size'] == 32
        assert isinstance(config_dict['optimizer'], dict)
        assert isinstance(config_dict['scheduler'], dict)
        assert isinstance(config_dict['loss'], dict)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'num_epochs': 50,
            'batch_size': 16,
            'optimizer': {
                'name': 'adam',
                'learning_rate': 1e-3
            },
            'scheduler': {
                'name': 'step',
                'warmup_steps': 500
            },
            'loss': {
                'price_loss_weight': 2.0,
                'quantiles': [0.25, 0.5, 0.75]
            }
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        assert config.num_epochs == 50
        assert config.batch_size == 16
        assert config.optimizer.name == 'adam'
        assert config.optimizer.learning_rate == 1e-3
        assert config.scheduler.name == 'step'
        assert config.scheduler.warmup_steps == 500
        assert config.loss.price_loss_weight == 2.0
        assert config.loss.quantiles == [0.25, 0.5, 0.75]
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = TrainingConfig(
            num_epochs=50,
            batch_size=16,
            experiment_name="test_experiment"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save(f.name)
            
            # Load and verify
            loaded_config = TrainingConfig.from_yaml(f.name)
            
            assert loaded_config.num_epochs == 50
            assert loaded_config.batch_size == 16
            assert loaded_config.experiment_name == "test_experiment"
            
            # Clean up
            Path(f.name).unlink()


class TestTrainingConfigValidator:
    """Test Pydantic validator."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config_data = {
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'gradient_clip': 1.0,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        }
        
        validator = TrainingConfigValidator(**config_data)
        assert validator.num_epochs == 100
        assert validator.batch_size == 32
    
    def test_invalid_epochs(self):
        """Test validation error for invalid epochs."""
        with pytest.raises(ValueError):
            TrainingConfigValidator(
                num_epochs=0,
                batch_size=32,
                learning_rate=1e-4,
                gradient_clip=1.0,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15
            )
    
    def test_invalid_splits(self):
        """Test validation error for invalid splits."""
        with pytest.raises(ValueError):
            TrainingConfigValidator(
                num_epochs=100,
                batch_size=32,
                learning_rate=1e-4,
                gradient_clip=1.0,
                train_split=0.8,
                val_split=0.15,
                test_split=0.15  # Sum = 1.1
            )


class TestConfigFactories:
    """Test configuration factory functions."""
    
    def test_create_default_config(self):
        """Test default configuration factory."""
        config = create_default_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.experiment_name == "transformer_training"
    
    def test_create_quick_test_config(self):
        """Test quick test configuration factory."""
        config = create_quick_test_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.num_epochs == 5
        assert config.batch_size == 16
        assert config.experiment_name == "quick_test"
        assert config.early_stopping_patience == 3
    
    def test_create_production_config(self):
        """Test production configuration factory."""
        config = create_production_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.num_epochs == 200
        assert config.batch_size == 64
        assert config.gradient_accumulation_steps == 2
        assert config.experiment_name == "production_training"
        assert config.early_stopping_patience == 20