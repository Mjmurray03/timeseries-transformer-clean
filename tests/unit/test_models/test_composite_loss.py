"""
Unit tests for CompositeLoss class.

Tests the multi-objective loss function that combines price, direction,
volatility, and quantile losses with specified weights.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.losses.composite_loss import CompositeLoss


class TestCompositeLoss:
    """Test suite for CompositeLoss"""
    
    @pytest.fixture
    def loss_function(self):
        """Create CompositeLoss instance for testing"""
        return CompositeLoss()
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample model predictions"""
        batch_size, horizon, n_quantiles = 4, 5, 5
        return {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1,
            'quantiles': torch.randn(batch_size, horizon, n_quantiles)
        }
    
    @pytest.fixture
    def sample_targets(self):
        """Generate sample target values"""
        batch_size, horizon = 4, 5
        return {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1
        }
    
    def test_initialization_default_weights(self):
        """Test CompositeLoss initializes with correct default weights"""
        loss_fn = CompositeLoss()
        
        # Check weights match requirements.md specification
        assert loss_fn.price_weight == 0.5
        assert loss_fn.direction_weight == 0.3
        assert loss_fn.volatility_weight == 0.1
        assert loss_fn.quantile_weight == 0.1
        
        # Check weights sum to 1.0
        total_weight = (
            loss_fn.price_weight + loss_fn.direction_weight + 
            loss_fn.volatility_weight + loss_fn.quantile_weight
        )
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_initialization_custom_weights(self):
        """Test CompositeLoss with custom weights"""
        custom_weights = {
            'price_weight': 0.4,
            'direction_weight': 0.4,
            'volatility_weight': 0.1,
            'quantile_weight': 0.1
        }
        
        loss_fn = CompositeLoss(**custom_weights)
        
        assert loss_fn.price_weight == 0.4
        assert loss_fn.direction_weight == 0.4
        assert loss_fn.volatility_weight == 0.1
        assert loss_fn.quantile_weight == 0.1
    
    def test_initialization_invalid_weights(self):
        """Test CompositeLoss raises error for invalid weights"""
        with pytest.raises(ValueError, match="Loss weights must sum to 1.0"):
            CompositeLoss(
                price_weight=0.5,
                direction_weight=0.5,
                volatility_weight=0.5,
                quantile_weight=0.5
            )
    
    def test_initialization_custom_quantiles(self):
        """Test CompositeLoss with custom quantile levels"""
        custom_quantiles = [0.25, 0.5, 0.75]
        loss_fn = CompositeLoss(quantile_levels=custom_quantiles)
        
        assert loss_fn.quantile_levels == custom_quantiles
    
    def test_forward_pass_basic(self, loss_function, sample_predictions, sample_targets):
        """Test basic forward pass returns loss and components"""
        total_loss, loss_components = loss_function(sample_predictions, sample_targets)
        
        # Check return types
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        
        # Check loss is scalar
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        
        # Check all loss components are present
        expected_components = ['price_loss', 'direction_loss', 'volatility_loss', 'quantile_loss', 'total_loss']
        for component in expected_components:
            assert component in loss_components
            assert isinstance(loss_components[component], float)
            assert loss_components[component] >= 0
    
    def test_forward_pass_shapes(self, loss_function):
        """Test forward pass with different input shapes"""
        # Test different batch sizes and horizons
        test_cases = [
            (1, 1),   # Single sample, single step
            (1, 5),   # Single sample, multi-step
            (8, 5),   # Batch, multi-step
            (16, 10)  # Large batch, longer horizon
        ]
        
        for batch_size, horizon in test_cases:
            predictions = {
                'price': torch.randn(batch_size, horizon),
                'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1,
                'quantiles': torch.randn(batch_size, horizon, 5)
            }
            targets = {
                'price': torch.randn(batch_size, horizon),
                'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1
            }
            
            total_loss, loss_components = loss_function(predictions, targets)
            
            assert total_loss.dim() == 0
            assert total_loss.item() >= 0
    
    def test_price_loss_component(self, loss_function, sample_predictions, sample_targets):
        """Test price loss component calculation"""
        total_loss, loss_components = loss_function(sample_predictions, sample_targets)
        
        # Calculate expected price loss manually
        expected_price_loss = torch.nn.functional.mse_loss(
            sample_predictions['price'], 
            sample_targets['price']
        ).item()
        
        # Check price loss component matches
        assert abs(loss_components['price_loss'] - expected_price_loss) < 1e-5
    
    def test_volatility_loss_component(self, loss_function, sample_predictions, sample_targets):
        """Test volatility loss component calculation"""
        total_loss, loss_components = loss_function(sample_predictions, sample_targets)
        
        # Calculate expected volatility loss manually
        expected_volatility_loss = torch.nn.functional.mse_loss(
            sample_predictions['volatility'], 
            sample_targets['volatility']
        ).item()
        
        # Check volatility loss component matches
        assert abs(loss_components['volatility_loss'] - expected_volatility_loss) < 1e-5
    
    def test_direction_loss_component(self, loss_function):
        """Test direction loss component with known directional patterns"""
        # Create predictions and targets with clear directional patterns
        batch_size, horizon = 2, 3
        
        # Upward trend in predictions and targets
        predictions = {
            'price': torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            'volatility': torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            'quantiles': torch.randn(batch_size, horizon, 5)
        }
        targets = {
            'price': torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            'volatility': torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        }
        
        total_loss, loss_components = loss_function(predictions, targets)
        
        # Direction loss should be low for perfect directional match
        assert loss_components['direction_loss'] < 1.0
    
    def test_quantile_loss_component(self, loss_function):
        """Test quantile loss component calculation"""
        batch_size, horizon = 2, 3
        
        # Create simple test case
        predictions = {
            'price': torch.ones(batch_size, horizon),
            'volatility': torch.ones(batch_size, horizon) * 0.1,
            'quantiles': torch.ones(batch_size, horizon, 5)
        }
        targets = {
            'price': torch.ones(batch_size, horizon),
            'volatility': torch.ones(batch_size, horizon) * 0.1
        }
        
        total_loss, loss_components = loss_function(predictions, targets)
        
        # Quantile loss should be present and non-negative
        assert 'quantile_loss' in loss_components
        assert loss_components['quantile_loss'] >= 0
    
    def test_loss_weighting(self):
        """Test that loss components are weighted correctly"""
        # Create loss function with known weights
        weights = {
            'price_weight': 0.4,
            'direction_weight': 0.3,
            'volatility_weight': 0.2,
            'quantile_weight': 0.1
        }
        loss_fn = CompositeLoss(**weights)
        
        # Create simple test data
        batch_size, horizon = 2, 3
        predictions = {
            'price': torch.ones(batch_size, horizon) * 2.0,
            'volatility': torch.ones(batch_size, horizon) * 0.2,
            'quantiles': torch.ones(batch_size, horizon, 5) * 2.0
        }
        targets = {
            'price': torch.ones(batch_size, horizon),
            'volatility': torch.ones(batch_size, horizon) * 0.1
        }
        
        total_loss, loss_components = loss_fn(predictions, targets)
        
        # Calculate expected total loss
        expected_total = (
            weights['price_weight'] * loss_components['price_loss'] +
            weights['direction_weight'] * loss_components['direction_loss'] +
            weights['volatility_weight'] * loss_components['volatility_loss'] +
            weights['quantile_weight'] * loss_components['quantile_loss']
        )
        
        assert abs(loss_components['total_loss'] - expected_total) < 1e-5
    
    def test_gradient_flow(self, loss_function, sample_predictions, sample_targets):
        """Test that gradients flow through all loss components"""
        # Enable gradients for predictions
        for key in sample_predictions:
            sample_predictions[key].requires_grad_(True)
        
        total_loss, _ = loss_function(sample_predictions, sample_targets)
        total_loss.backward()
        
        # Check gradients exist for all prediction tensors
        for key, tensor in sample_predictions.items():
            assert tensor.grad is not None
            assert not torch.isnan(tensor.grad).any()
            assert not torch.isinf(tensor.grad).any()
    
    def test_get_weights(self, loss_function):
        """Test get_weights method returns correct weights"""
        weights = loss_function.get_weights()
        
        expected_weights = {
            'price_weight': 0.5,
            'direction_weight': 0.3,
            'volatility_weight': 0.1,
            'quantile_weight': 0.1
        }
        
        assert weights == expected_weights
    
    def test_update_weights_valid(self, loss_function):
        """Test update_weights method with valid weights"""
        new_weights = {
            'price_weight': 0.6,
            'direction_weight': 0.2,
            'volatility_weight': 0.1,
            'quantile_weight': 0.1
        }
        
        loss_function.update_weights(**new_weights)
        
        assert loss_function.price_weight == 0.6
        assert loss_function.direction_weight == 0.2
        assert loss_function.volatility_weight == 0.1
        assert loss_function.quantile_weight == 0.1
    
    def test_update_weights_invalid_sum(self, loss_function):
        """Test update_weights raises error for invalid weight sum"""
        with pytest.raises(ValueError, match="Updated weights must sum to 1.0"):
            loss_function.update_weights(
                price_weight=0.8,
                direction_weight=0.8
            )
    
    def test_update_weights_unknown_parameter(self, loss_function):
        """Test update_weights raises error for unknown parameter"""
        with pytest.raises(ValueError, match="Unknown weight parameter"):
            loss_function.update_weights(unknown_weight=0.5)
    
    def test_device_compatibility(self, loss_function):
        """Test loss function works on different devices"""
        batch_size, horizon = 2, 3
        
        # Test on CPU
        predictions_cpu = {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1,
            'quantiles': torch.randn(batch_size, horizon, 5)
        }
        targets_cpu = {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1
        }
        
        total_loss_cpu, _ = loss_function(predictions_cpu, targets_cpu)
        assert total_loss_cpu.device.type == 'cpu'
        
        # Test on GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            loss_function_gpu = loss_function.to(device)
            
            predictions_gpu = {k: v.to(device) for k, v in predictions_cpu.items()}
            targets_gpu = {k: v.to(device) for k, v in targets_cpu.items()}
            
            total_loss_gpu, _ = loss_function_gpu(predictions_gpu, targets_gpu)
            assert total_loss_gpu.device.type == 'cuda'
    
    def test_numerical_stability(self, loss_function):
        """Test loss function handles edge cases numerically"""
        batch_size, horizon = 2, 3
        
        # Test with very small values
        predictions_small = {
            'price': torch.ones(batch_size, horizon) * 1e-8,
            'volatility': torch.ones(batch_size, horizon) * 1e-8,
            'quantiles': torch.ones(batch_size, horizon, 5) * 1e-8
        }
        targets_small = {
            'price': torch.ones(batch_size, horizon) * 1e-8,
            'volatility': torch.ones(batch_size, horizon) * 1e-8
        }
        
        total_loss, loss_components = loss_function(predictions_small, targets_small)
        
        assert torch.isfinite(total_loss)
        for component_loss in loss_components.values():
            assert np.isfinite(component_loss)
        
        # Test with larger values
        predictions_large = {
            'price': torch.ones(batch_size, horizon) * 1e3,
            'volatility': torch.ones(batch_size, horizon) * 1e2,
            'quantiles': torch.ones(batch_size, horizon, 5) * 1e3
        }
        targets_large = {
            'price': torch.ones(batch_size, horizon) * 1e3,
            'volatility': torch.ones(batch_size, horizon) * 1e2
        }
        
        total_loss, loss_components = loss_function(predictions_large, targets_large)
        
        assert torch.isfinite(total_loss)
        for component_loss in loss_components.values():
            assert np.isfinite(component_loss)
    
    def test_batch_consistency(self, loss_function):
        """Test loss is consistent across different batch sizes"""
        horizon = 5
        
        # Single sample
        single_predictions = {
            'price': torch.randn(1, horizon),
            'volatility': torch.abs(torch.randn(1, horizon)) * 0.1,
            'quantiles': torch.randn(1, horizon, 5)
        }
        single_targets = {
            'price': torch.randn(1, horizon),
            'volatility': torch.abs(torch.randn(1, horizon)) * 0.1
        }
        
        # Batch with same sample repeated
        batch_predictions = {k: v.repeat(4, 1, 1) if v.dim() == 3 else v.repeat(4, 1) 
                           for k, v in single_predictions.items()}
        batch_targets = {k: v.repeat(4, 1) for k, v in single_targets.items()}
        
        single_loss, _ = loss_function(single_predictions, single_targets)
        batch_loss, _ = loss_function(batch_predictions, batch_targets)
        
        # Losses should be approximately equal (batch loss is average)
        assert abs(single_loss.item() - batch_loss.item()) < 1e-5
    
    @pytest.mark.parametrize("quantile_levels", [
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
        [0.05, 0.25, 0.5, 0.75, 0.95]
    ])
    def test_different_quantile_levels(self, quantile_levels):
        """Test loss function with different quantile level configurations"""
        loss_fn = CompositeLoss(quantile_levels=quantile_levels)
        
        batch_size, horizon = 2, 3
        n_quantiles = len(quantile_levels)
        
        predictions = {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1,
            'quantiles': torch.randn(batch_size, horizon, n_quantiles)
        }
        targets = {
            'price': torch.randn(batch_size, horizon),
            'volatility': torch.abs(torch.randn(batch_size, horizon)) * 0.1
        }
        
        total_loss, loss_components = loss_fn(predictions, targets)
        
        assert torch.isfinite(total_loss)
        assert loss_components['quantile_loss'] >= 0
    
    def test_zero_loss_perfect_predictions(self):
        """Test loss approaches zero for perfect predictions"""
        loss_fn = CompositeLoss()
        
        batch_size, horizon = 2, 3
        
        # Perfect predictions (identical to targets)
        perfect_price = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        perfect_volatility = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        
        predictions = {
            'price': perfect_price.clone(),
            'volatility': perfect_volatility.clone(),
            'quantiles': perfect_price.unsqueeze(-1).repeat(1, 1, 5)  # All quantiles = price
        }
        targets = {
            'price': perfect_price.clone(),
            'volatility': perfect_volatility.clone()
        }
        
        total_loss, loss_components = loss_fn(predictions, targets)
        
        # Price and volatility losses should be exactly zero
        assert abs(loss_components['price_loss']) < 1e-6
        assert abs(loss_components['volatility_loss']) < 1e-6
        
        # Total loss should be very small (only direction and quantile components)
        assert total_loss.item() < 1.0