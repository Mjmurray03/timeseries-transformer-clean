"""
Comprehensive tests for TrainingConfig parameter compatibility and from_args method.
Tests verify that the extended TrainingConfig properly handles all parameter scenarios
while maintaining 100% backward compatibility.
"""

import pytest
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.training_config import (
    TrainingConfig, 
    OptimizerConfig, 
    SchedulerConfig, 
    LossConfig,
    create_default_config,
    create_quick_test_config,
    create_production_config,
    get_parameter_mapping_documentation
)


class TestTrainingConfigParameterCompatibility:
    """Comprehensive test suite for TrainingConfig parameter handling."""
    
    def test_from_args_all_parameters_provided(self):
        """Test from_args with comprehensive parameter set."""
        # Test case 1: Maximum parameter coverage
        all_args = {
            # Basic training parameters
            'epochs': 150,
            'batch_size': 64,
            'gradient_accumulation_steps': 2,
            'gradient_clip': 1.5,
            
            # Mixed precision and device
            'use_amp': True,
            'device': 'cuda',
            'num_workers': 8,
            'pin_memory': True,
            
            # Checkpointing
            'save_every': 5,
            'checkpoint_dir': '/tmp/checkpoints',
            'save_best_only': False,
            
            # Early stopping
            'early_stopping_patience': 20,
            'early_stopping_min_delta': 0.001,
            
            # Experiment tracking
            'experiment_name': 'full_test_run',
            'project_name': 'test_project',
            'log_every': 50,
            
            # Validation
            'val_every': 2,
            'val_metric': 'accuracy',
            
            # Data splits
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            
            # Reproducibility
            'seed': 12345,
            'deterministic': False,
            
            # Optimizer parameters
            'learning_rate': 0.0005,
            'weight_decay': 0.01,
            'optimizer_name': 'adamw',
            'amsgrad': True,
            
            # Scheduler parameters
            'scheduler_name': 'cosine',
            'warmup_steps': 2000,
            'max_steps': 20000,
            'min_lr': 1e-7,
            'patience': 15,
            'factor': 0.8
        }
        
        config = TrainingConfig.from_args(all_args)
        
        # Verify all parameters were set correctly
        assert config.num_epochs == 150
        assert config.batch_size == 64
        assert config.gradient_accumulation_steps == 2
        assert config.gradient_clip == 1.5
        assert config.use_amp == True
        assert config.device == 'cuda'
        assert config.num_workers == 8
        assert config.pin_memory == True
        assert config.save_every == 5
        assert config.checkpoint_dir == '/tmp/checkpoints'
        assert config.save_best_only == False
        assert config.early_stopping_patience == 20
        assert config.early_stopping_min_delta == 0.001
        assert config.experiment_name == 'full_test_run'
        assert config.project_name == 'test_project'
        assert config.log_every == 50
        assert config.val_every == 2
        assert config.val_metric == 'accuracy'
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.seed == 12345
        assert config.deterministic == False
        assert config.optimizer.learning_rate == 0.0005
        assert config.optimizer.weight_decay == 0.01
        assert config.optimizer.name == 'adamw'
        assert config.optimizer.amsgrad == True
        assert config.scheduler.name == 'cosine'
        assert config.scheduler.warmup_steps == 2000
        assert config.scheduler.max_steps == 20000
        assert config.scheduler.min_lr == 1e-7
        assert config.scheduler.patience == 15
        assert config.scheduler.factor == 0.8
    
    @pytest.mark.parametrize("parameter_set", [
        # Test case 2: Minimal required parameters
        {'epochs': 10},
        
        # Test case 3: Basic training setup
        {'epochs': 50, 'batch_size': 32, 'learning_rate': 0.001},
        
        # Test case 4: GPU training setup
        {'device': 'cuda', 'use_amp': True, 'batch_size': 64},
        
        # Test case 5: CPU training setup
        {'device': 'cpu', 'use_amp': False, 'num_workers': 0},
        
        # Test case 6: Quick experiment
        {'epochs': 5, 'batch_size': 16, 'experiment_name': 'quick_test'},
        
        # Test case 7: Production setup
        {'epochs': 200, 'batch_size': 128, 'early_stopping_patience': 30},
        
        # Test case 8: Custom optimizer
        {'learning_rate': 0.0001, 'weight_decay': 0.1, 'optimizer_name': 'adamw'},
        
        # Test case 9: Custom scheduler
        {'scheduler_name': 'plateau', 'patience': 5, 'factor': 0.5},
        
        # Test case 10: Reproducible setup
        {'seed': 42, 'deterministic': True},
        
        # Test case 11: Custom data splits
        {'train_split': 0.7, 'val_split': 0.2, 'test_split': 0.1},
        
        # Test case 12: Extensive logging
        {'log_every': 10, 'val_every': 1, 'save_every': 2},
        
        # Test case 13: High-performance setup
        {'num_workers': 16, 'pin_memory': True, 'gradient_accumulation_steps': 4},
        
        # Test case 14: Conservative training
        {'gradient_clip': 0.5, 'early_stopping_min_delta': 1e-6},
        
        # Test case 15: Dash-style arguments
        {'batch-size': 32, 'learning-rate': 0.001, 'use-amp': True},
        
        # Test case 16: Mixed argument styles
        {'epochs': 100, 'batch-size': 64, 'learning_rate': 0.0005},
        
        # Test case 17: Boolean string conversion
        {'use_amp': 'true', 'deterministic': 'false', 'save_best_only': '1'},
        
        # Test case 18: Short aliases
        {'lr': 0.001, 'epochs': 25},
        
        # Test case 19: Checkpoint configuration
        {'checkpoint_dir': './models', 'save_best_only': True, 'save_every': 10},
        
        # Test case 20: Advanced scheduler configuration
        {'scheduler_name': 'cosine', 'warmup_steps': 1000, 'max_steps': 50000, 'min_lr': 1e-8},
        
        # Test case 21: Edge case values
        {'epochs': 1, 'batch_size': 1, 'learning_rate': 1e-8},
        
        # Test case 22: Large values
        {'epochs': 1000, 'batch_size': 512, 'max_steps': 100000}
    ])
    def test_from_args_parameter_combinations(self, parameter_set):
        """Test from_args with 22 different parameter combinations."""
        config = TrainingConfig.from_args(parameter_set)
        
        # Verify config was created successfully
        assert isinstance(config, TrainingConfig)
        
        # Verify validation passes
        config.validate()
        
        # Verify specific parameters were set
        for key, expected_value in parameter_set.items():
            # Handle parameter mapping
            if key == 'epochs':
                assert config.num_epochs == expected_value
            elif key == 'batch-size':
                assert config.batch_size == expected_value
            elif key == 'learning-rate' or key == 'lr':
                assert config.optimizer.learning_rate == expected_value
            elif key == 'use-amp':
                expected_bool = expected_value
                if isinstance(expected_value, str):
                    expected_bool = expected_value.lower() in ('true', '1', 'yes', 'on')
                assert config.use_amp == expected_bool
            elif key == 'optimizer_name':
                assert config.optimizer.name == expected_value
            elif key == 'scheduler_name':
                assert config.scheduler.name == expected_value
            elif hasattr(config, key):
                actual_value = getattr(config, key)
                # Handle boolean string conversion for comparison
                if isinstance(expected_value, str) and key in ['deterministic', 'save_best_only', 'use_amp']:
                    expected_bool = expected_value.lower() in ('true', '1', 'yes', 'on')
                    assert actual_value == expected_bool
                else:
                    assert actual_value == expected_value
    
    def test_from_args_with_argparse_namespace(self):
        """Test from_args with actual argparse.Namespace object."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--use-amp', action='store_true')
        parser.add_argument('--experiment-name', type=str, default='test_run')
        
        # Test with command-line style arguments
        args = parser.parse_args([
            '--epochs', '75',
            '--batch-size', '48', 
            '--learning-rate', '0.0008',
            '--device', 'cuda',
            '--use-amp',
            '--experiment-name', 'argparse_test'
        ])
        
        config = TrainingConfig.from_args(args)
        
        assert config.num_epochs == 75
        assert config.batch_size == 48
        assert config.optimizer.learning_rate == 0.0008
        assert config.device == 'cuda'
        assert config.use_amp == True
        assert config.experiment_name == 'argparse_test'
    
    def test_backward_compatibility_original_initialization(self):
        """Test that original TrainingConfig initialization still works."""
        # Test 1: Default initialization
        config1 = TrainingConfig()
        assert config1.num_epochs == 100
        assert config1.batch_size == 32
        assert config1.optimizer.learning_rate == 1e-4
        
        # Test 2: Manual parameter initialization
        config2 = TrainingConfig(
            num_epochs=50,
            batch_size=64,
            device='cpu',
            experiment_name='manual_test'
        )
        assert config2.num_epochs == 50
        assert config2.batch_size == 64
        assert config2.device == 'cpu'
        assert config2.experiment_name == 'manual_test'
        
        # Test 3: With nested configs
        optimizer = OptimizerConfig(learning_rate=0.002, weight_decay=0.05)
        scheduler = SchedulerConfig(name='plateau', patience=5)
        loss = LossConfig(price_loss_weight=2.0)
        
        config3 = TrainingConfig(
            num_epochs=25,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss
        )
        assert config3.num_epochs == 25
        assert config3.optimizer.learning_rate == 0.002
        assert config3.optimizer.weight_decay == 0.05
        assert config3.scheduler.name == 'plateau'
        assert config3.scheduler.patience == 5
        assert config3.loss.price_loss_weight == 2.0
    
    def test_backward_compatibility_factory_functions(self):
        """Test that existing factory functions still work."""
        # Test default config
        default_config = create_default_config()
        assert isinstance(default_config, TrainingConfig)
        assert default_config.num_epochs == 100
        assert default_config.batch_size == 32
        
        # Test quick test config
        quick_config = create_quick_test_config()
        assert isinstance(quick_config, TrainingConfig)
        assert quick_config.num_epochs == 5
        assert quick_config.batch_size == 16
        assert quick_config.experiment_name == 'quick_test'
        
        # Test production config
        prod_config = create_production_config()
        assert isinstance(prod_config, TrainingConfig)
        assert prod_config.num_epochs == 200
        assert prod_config.batch_size == 64
        assert prod_config.experiment_name == 'production_training'
    
    def test_backward_compatibility_existing_methods(self):
        """Test that existing methods (from_yaml, from_dict, to_dict) still work."""
        # Test from_dict
        config_dict = {
            'num_epochs': 80,
            'batch_size': 48,
            'optimizer': {
                'learning_rate': 0.0015,
                'weight_decay': 0.02
            },
            'scheduler': {
                'name': 'cosine',
                'max_steps': 8000
            }
        }
        
        config_from_dict = TrainingConfig.from_dict(config_dict)
        assert config_from_dict.num_epochs == 80
        assert config_from_dict.batch_size == 48
        assert config_from_dict.optimizer.learning_rate == 0.0015
        assert config_from_dict.scheduler.name == 'cosine'
        
        # Test to_dict
        config_dict_output = config_from_dict.to_dict()
        assert config_dict_output['num_epochs'] == 80
        assert config_dict_output['batch_size'] == 48
        assert config_dict_output['optimizer']['learning_rate'] == 0.0015
        
        # Test from_yaml with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_file = f.name
        
        try:
            config_from_yaml = TrainingConfig.from_yaml(yaml_file)
            assert config_from_yaml.num_epochs == 80
            assert config_from_yaml.batch_size == 48
        finally:
            Path(yaml_file).unlink()
    
    def test_integration_with_training_orchestrator(self):
        """Test that configs created via from_args work with TrainingOrchestrator."""
        from src.models.timeseries_transformer import TimeSeriesTransformer
        
        # Create config using from_args
        config_args = {
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 0.001,
            'device': 'cpu',
            'use_amp': False,
            'gradient_clip': 1.0,
            'experiment_name': 'orchestrator_test',
            'early_stopping_patience': 3
        }
        
        config = TrainingConfig.from_args(config_args)
        
        # Verify all parameters required by TrainingOrchestrator are present
        required_attributes = [
            'device', 'use_amp', 'experiment_name', 'project_name',
            'early_stopping_patience', 'early_stopping_min_delta',
            'checkpoint_dir', 'save_best_only', 'deterministic', 'seed',
            'num_epochs', 'val_every', 'log_every', 'gradient_accumulation_steps',
            'gradient_clip'
        ]
        
        for attr in required_attributes:
            assert hasattr(config, attr), f"Missing required attribute: {attr}"
            value = getattr(config, attr)
            assert value is not None, f"Attribute {attr} is None"
        
        # Test nested config attributes
        assert hasattr(config, 'optimizer') and hasattr(config.optimizer, 'learning_rate')
        assert hasattr(config, 'scheduler') and hasattr(config.scheduler, 'name')
        assert hasattr(config, 'loss') and hasattr(config.loss, 'quantiles')
        
        # Create a simple model to test compatibility
        model = TimeSeriesTransformer(
            input_dim=10,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            forecast_horizon=5
        )
        
        # Verify config can be used to initialize training components
        # (without actually creating TrainingOrchestrator to avoid heavy dependencies)
        assert config.device in ['cpu', 'cuda', 'mps']
        assert isinstance(config.use_amp, bool)
        assert config.num_epochs > 0
        assert config.batch_size > 0
        assert config.optimizer.learning_rate > 0
        
        # Test config validation passes
        config.validate()
    
    @pytest.mark.parametrize("invalid_params,expected_error", [
        # Test invalid parameter types
        ({'epochs': 'not_a_number'}, ValueError),
        ({'batch_size': -5}, ValueError),
        ({'learning_rate': 0}, ValueError),
        ({'learning_rate': -0.001}, ValueError),
        ({'device': 'invalid_device'}, ValueError),
        ({'gradient_clip': -1.0}, ValueError),
        ({'train_split': 1.5}, ValueError),
        ({'val_split': -0.1}, ValueError),
        ({'test_split': 0}, ValueError),
        ({'early_stopping_patience': 0}, ValueError),
        ({'save_every': -1}, ValueError),
        ({'log_every': 0}, ValueError),
        ({'num_workers': -1}, ValueError),
        ({'warmup_steps': -100}, ValueError),
        ({'max_steps': 0}, ValueError),
        ({'min_lr': -1e-6}, ValueError),
        ({'patience': 0}, ValueError),
        ({'factor': 0}, ValueError),
        ({'seed': -1}, ValueError),
        # Test invalid input types to from_args
        ('not_dict_or_namespace', TypeError),
        (123, TypeError),
        ([1, 2, 3], TypeError),
    ])
    def test_parameter_validation_errors(self, invalid_params, expected_error):
        """Test that invalid parameters raise appropriate errors with clear messages."""
        with pytest.raises(expected_error) as exc_info:
            if isinstance(invalid_params, dict):
                TrainingConfig.from_args(invalid_params)
            else:
                TrainingConfig.from_args(invalid_params)
        
        # Verify error message is informative
        error_message = str(exc_info.value)
        if isinstance(invalid_params, dict):
            param_name = list(invalid_params.keys())[0]
            if param_name in ['epochs', 'batch_size', 'learning_rate', 'device']:
                assert param_name in error_message or 'Invalid value' in error_message
    
    def test_exact_error_messages(self):
        """Test exact error messages for specific validation failures."""
        # Test negative epochs
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig.from_args({'epochs': -10})
        assert "num_epochs must be positive" in str(exc_info.value)
        
        # Test zero batch size
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig.from_args({'batch_size': 0})
        assert "batch_size must be positive" in str(exc_info.value)
        
        # Test negative learning rate
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig.from_args({'learning_rate': -0.001})
        assert "learning_rate must be positive" in str(exc_info.value)
        
        # Test invalid device
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig.from_args({'device': 'tpu'})
        assert "device must be one of ['cpu', 'cuda', 'mps']" in str(exc_info.value)
        
        # Test invalid train split
        with pytest.raises(ValueError) as exc_info:
            TrainingConfig.from_args({'train_split': 1.2})
        assert "train_split must be between 0 and 1" in str(exc_info.value)
        
        # Test wrong input type
        with pytest.raises(TypeError) as exc_info:
            TrainingConfig.from_args("invalid_input")
        assert "Expected argparse.Namespace or dict" in str(exc_info.value)
    
    def test_default_values_documentation(self):
        """Test and document all default values."""
        config = TrainingConfig()
        
        # Document all defaults with assertions
        defaults = {
            'num_epochs': 100,
            'batch_size': 32,
            'gradient_accumulation_steps': 1,
            'gradient_clip': 1.0,
            'use_amp': True,
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'save_every': 10,
            'checkpoint_dir': 'models/checkpoints',
            'save_best_only': True,
            'early_stopping_patience': 10,
            'early_stopping_min_delta': 1e-4,
            'experiment_name': 'transformer_training',
            'project_name': 'timeseries-transformer',
            'log_every': 100,
            'val_every': 1,
            'val_metric': 'loss',
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'seed': 42,
            'deterministic': True
        }
        
        for param, expected_default in defaults.items():
            actual_value = getattr(config, param)
            assert actual_value == expected_default, \
                f"Default for {param}: expected {expected_default}, got {actual_value}"
        
        # Test optimizer defaults
        optimizer_defaults = {
            'name': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }
        
        for param, expected_default in optimizer_defaults.items():
            actual_value = getattr(config.optimizer, param)
            assert actual_value == expected_default, \
                f"Optimizer default for {param}: expected {expected_default}, got {actual_value}"
        
        # Test scheduler defaults
        scheduler_defaults = {
            'name': 'cosine',
            'warmup_steps': 1000,
            'max_steps': 10000,
            'min_lr': 1e-6,
            'patience': 10,
            'factor': 0.5
        }
        
        for param, expected_default in scheduler_defaults.items():
            actual_value = getattr(config.scheduler, param)
            assert actual_value == expected_default, \
                f"Scheduler default for {param}: expected {expected_default}, got {actual_value}"
        
        # Test loss defaults
        assert config.loss.price_loss_weight == 1.0
        assert config.loss.direction_loss_weight == 0.5
        assert config.loss.volatility_loss_weight == 0.3
        assert config.loss.quantile_loss_weight == 0.2
        assert config.loss.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
    
    def test_defaults_prevent_training_failures(self):
        """Test that default values don't cause training failures."""
        config = create_default_config()
        
        # Verify splits sum to 1.0
        total_split = config.train_split + config.val_split + config.test_split
        assert abs(total_split - 1.0) < 1e-6
        
        # Verify all critical parameters are positive
        assert config.num_epochs > 0
        assert config.batch_size > 0
        assert config.optimizer.learning_rate > 0
        assert config.gradient_clip > 0
        assert config.early_stopping_patience > 0
        assert config.save_every > 0
        assert config.log_every > 0
        assert config.val_every > 0
        
        # Verify device is valid
        assert config.device in ['cpu', 'cuda', 'mps']
        
        # Verify quantiles are valid
        assert all(0 < q < 1 for q in config.loss.quantiles)
        
        # Test validation passes
        config.validate()
    
    def test_parameter_mapping_documentation(self):
        """Test parameter mapping documentation function."""
        docs = get_parameter_mapping_documentation()
        
        # Verify documentation is comprehensive
        assert isinstance(docs, str)
        assert len(docs) > 1000  # Should be substantial documentation
        
        # Verify key sections are present
        required_sections = [
            'BASIC TRAINING PARAMETERS',
            'DEVICE AND PERFORMANCE', 
            'OPTIMIZER PARAMETERS',
            'SCHEDULER PARAMETERS',
            'USAGE EXAMPLES',
            'ERROR HANDLING',
            'BACKWARD COMPATIBILITY'
        ]
        
        for section in required_sections:
            assert section in docs, f"Missing documentation section: {section}"
        
        # Verify key parameter mappings are documented
        key_mappings = [
            '--epochs',
            '--batch-size',
            '--learning-rate',
            '--device',
            '--use-amp'
        ]
        
        for mapping in key_mappings:
            assert mapping in docs, f"Missing parameter mapping: {mapping}"
    
    def test_boolean_conversion_comprehensive(self):
        """Test comprehensive boolean conversion scenarios."""
        # Test all true variations
        true_values = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES', 'on', 'On', 'ON']
        for true_val in true_values:
            config = TrainingConfig.from_args({'use_amp': true_val})
            assert config.use_amp == True, f"Failed to convert '{true_val}' to True"
        
        # Test all false variations
        false_values = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', 'off', 'Off', 'OFF']
        for false_val in false_values:
            config = TrainingConfig.from_args({'use_amp': false_val})
            assert config.use_amp == False, f"Failed to convert '{false_val}' to False"
        
        # Test native boolean values
        config_true = TrainingConfig.from_args({'use_amp': True})
        assert config_true.use_amp == True
        
        config_false = TrainingConfig.from_args({'use_amp': False})
        assert config_false.use_amp == False
    
    def test_parameter_name_variations(self):
        """Test that both underscore and dash variations work."""
        # Test underscore versions
        config_underscore = TrainingConfig.from_args({
            'num_epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_amp': True,
            'early_stopping_patience': 5
        })
        
        # Test dash versions
        config_dash = TrainingConfig.from_args({
            'epochs': 50,  # Alternative name
            'batch-size': 32,
            'learning-rate': 0.001,
            'use-amp': True,
            'early-stopping-patience': 5
        })
        
        # Verify both produce identical results
        assert config_underscore.num_epochs == config_dash.num_epochs
        assert config_underscore.batch_size == config_dash.batch_size
        assert config_underscore.optimizer.learning_rate == config_dash.optimizer.learning_rate
        assert config_underscore.use_amp == config_dash.use_amp
        assert config_underscore.early_stopping_patience == config_dash.early_stopping_patience
    
    def test_none_values_handling(self):
        """Test that None values are properly ignored."""
        args_with_none = {
            'epochs': 100,
            'batch_size': None,  # Should be ignored
            'learning_rate': 0.001,
            'device': None,  # Should be ignored
            'experiment_name': 'test_none'
        }
        
        config = TrainingConfig.from_args(args_with_none)
        
        # Verify provided values were set
        assert config.num_epochs == 100
        assert config.optimizer.learning_rate == 0.001
        assert config.experiment_name == 'test_none'
        
        # Verify None values resulted in defaults
        assert config.batch_size == 32  # Default
        assert config.device == 'cuda'  # Default
    
    def test_edge_case_values(self):
        """Test edge case parameter values."""
        edge_cases = [
            # Minimum valid values
            {'epochs': 1, 'batch_size': 1, 'learning_rate': 1e-10},
            
            # Maximum reasonable values
            {'epochs': 10000, 'batch_size': 1024, 'max_steps': 1000000},
            
            # Precision edge cases
            {'learning_rate': 1e-8, 'min_lr': 1e-12, 'early_stopping_min_delta': 1e-10},
            
            # Split edge cases
            {'train_split': 0.001, 'val_split': 0.001, 'test_split': 0.998},
        ]
        
        for edge_case in edge_cases:
            config = TrainingConfig.from_args(edge_case)
            config.validate()  # Should not raise any errors
    
    def test_silent_failure_prevention(self):
        """Test that no silent failures occur - all errors are caught and reported."""
        # Test that invalid configurations raise exceptions rather than silently failing
        
        # This should raise an exception, not silently fail
        with pytest.raises(ValueError):
            config = TrainingConfig.from_args({'epochs': -1})
        
        # This should raise an exception, not silently fail
        with pytest.raises(ValueError):
            config = TrainingConfig.from_args({'learning_rate': 0})
        
        # This should raise an exception, not silently fail
        with pytest.raises(ValueError):
            config = TrainingConfig.from_args({'device': 'nonexistent_device'})
        
        # Test that data split validation catches invalid combinations
        with pytest.raises(ValueError):
            # These splits don't sum to 1.0
            config = TrainingConfig(train_split=0.8, val_split=0.8, test_split=0.8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])