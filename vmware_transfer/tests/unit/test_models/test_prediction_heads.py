"""
Unit tests for prediction head components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.components.prediction_heads import (
    PricePredictionHead,
    VolatilityPredictionHead,
    QuantileRegressionHead,
    DirectionalPredictionHead,
    MultiTaskPredictionHead,
    ResidualBlock,
    create_prediction_head
)


class TestPricePredictionHead:
    """Test suite for PricePredictionHead"""
    
    @pytest.fixture
    def price_head(self):
        """Create price prediction head instance"""
        return PricePredictionHead(
            d_model=256, 
            forecast_horizon=5, 
            hidden_dim=128, 
            num_layers=2, 
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input_2d(self):
        """Generate 2D sample input tensor"""
        return torch.randn(2, 256)  # (batch_size, d_model)
    
    @pytest.fixture
    def sample_input_3d(self):
        """Generate 3D sample input tensor"""
        return torch.randn(2, 60, 256)  # (batch_size, seq_len, d_model)
    
    def test_initialization(self, price_head):
        """Test price head initializes correctly"""
        assert price_head.d_model == 256
        assert price_head.forecast_horizon == 5
        assert price_head.hidden_dim == 128
        assert price_head.num_layers == 2
        assert price_head.dropout == 0.1
        assert price_head.predict_changes == True
        
        # Check network exists
        assert hasattr(price_head, 'network')
        assert isinstance(price_head.network, nn.Sequential)
    
    def test_forward_pass_2d(self, price_head, sample_input_2d):
        """Test forward pass with 2D input"""
        output = price_head(sample_input_2d)
        
        assert output.shape == (2, 5)  # (batch_size, forecast_horizon)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_3d(self, price_head, sample_input_3d):
        """Test forward pass with 3D input (takes last time step)"""
        output = price_head(sample_input_3d)
        
        assert output.shape == (2, 5)  # (batch_size, forecast_horizon)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_configurations(self):
        """Test with different configuration parameters"""
        head = PricePredictionHead(
            d_model=128,
            forecast_horizon=10,
            hidden_dim=64,
            num_layers=3,
            activation="relu",
            predict_changes=False
        )
        
        input_tensor = torch.randn(4, 128)
        output = head(input_tensor)
        
        assert output.shape == (4, 10)
        assert head.predict_changes == False
    
    def test_gradient_flow(self, price_head, sample_input_2d):
        """Test gradients flow through price head"""
        sample_input_2d.requires_grad_(True)
        output = price_head(sample_input_2d)
        loss = output.sum()
        loss.backward()
        
        assert sample_input_2d.grad is not None
        assert not torch.isnan(sample_input_2d.grad).any()
        
        # Check gradients for network parameters
        for param in price_head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_invalid_activation(self):
        """Test error handling for invalid activation"""
        with pytest.raises(ValueError):
            PricePredictionHead(activation="invalid_activation")


class TestVolatilityPredictionHead:
    """Test suite for VolatilityPredictionHead"""
    
    @pytest.fixture
    def volatility_head(self):
        """Create volatility prediction head instance"""
        return VolatilityPredictionHead(
            d_model=256,
            forecast_horizon=5,
            hidden_dim=128,
            dropout=0.1,
            output_activation="softplus"
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 256)
    
    def test_initialization(self, volatility_head):
        """Test volatility head initializes correctly"""
        assert volatility_head.d_model == 256
        assert volatility_head.forecast_horizon == 5
        assert volatility_head.hidden_dim == 128
        assert isinstance(volatility_head.output_activation, nn.Softplus)
    
    def test_forward_pass(self, volatility_head, sample_input):
        """Test forward pass produces positive volatility"""
        output = volatility_head(sample_input)
        
        assert output.shape == (2, 5)
        assert (output >= 0).all()  # Volatility must be non-negative
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_output_activations(self):
        """Test different output activation functions"""
        activations = ["softplus", "exp", "relu"]
        
        for activation in activations:
            head = VolatilityPredictionHead(output_activation=activation)
            input_tensor = torch.randn(2, 256)
            output = head(input_tensor)
            
            assert output.shape == (2, 5)
            assert (output >= 0).all()  # All should produce non-negative outputs
    
    def test_invalid_output_activation(self):
        """Test error handling for invalid output activation"""
        with pytest.raises(ValueError):
            VolatilityPredictionHead(output_activation="invalid")
    
    def test_gradient_flow(self, volatility_head, sample_input):
        """Test gradients flow through volatility head"""
        sample_input.requires_grad_(True)
        output = volatility_head(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestQuantileRegressionHead:
    """Test suite for QuantileRegressionHead"""
    
    @pytest.fixture
    def quantile_head(self):
        """Create quantile regression head instance"""
        return QuantileRegressionHead(
            d_model=256,
            forecast_horizon=5,
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            hidden_dim=128,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 256)
    
    def test_initialization(self, quantile_head):
        """Test quantile head initializes correctly"""
        assert quantile_head.d_model == 256
        assert quantile_head.forecast_horizon == 5
        assert quantile_head.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert quantile_head.num_quantiles == 5
        
        # Check quantile heads
        assert len(quantile_head.quantile_heads) == 5
        for head in quantile_head.quantile_heads:
            assert isinstance(head, nn.Sequential)
    
    def test_forward_pass(self, quantile_head, sample_input):
        """Test forward pass produces correct quantile predictions"""
        output = quantile_head(sample_input)
        
        assert output.shape == (2, 5, 5)  # (batch_size, forecast_horizon, num_quantiles)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_quantile_ordering(self, quantile_head, sample_input):
        """Test that quantile predictions are monotonically ordered"""
        quantile_head.eval()
        
        with torch.no_grad():
            output = quantile_head(sample_input)
        
        # Check monotonic ordering along quantile dimension
        for b in range(output.shape[0]):
            for t in range(output.shape[1]):
                quantile_values = output[b, t, :]
                # Check if sorted (monotonic)
                assert torch.all(quantile_values[:-1] <= quantile_values[1:])
    
    def test_different_quantiles(self):
        """Test with different quantile specifications"""
        custom_quantiles = [0.05, 0.5, 0.95]
        head = QuantileRegressionHead(
            d_model=256,
            forecast_horizon=3,
            quantiles=custom_quantiles
        )
        
        input_tensor = torch.randn(2, 256)
        output = head(input_tensor)
        
        assert output.shape == (2, 3, 3)  # 3 quantiles
        assert head.quantiles == custom_quantiles
    
    def test_gradient_flow(self, quantile_head, sample_input):
        """Test gradients flow through all quantile heads"""
        sample_input.requires_grad_(True)
        output = quantile_head(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        
        # Check gradients for all quantile heads
        for i, head in enumerate(quantile_head.quantile_heads):
            for param in head.parameters():
                assert param.grad is not None, f"No gradient for quantile head {i}"


class TestDirectionalPredictionHead:
    """Test suite for DirectionalPredictionHead"""
    
    @pytest.fixture
    def direction_head(self):
        """Create directional prediction head instance"""
        return DirectionalPredictionHead(
            d_model=256,
            forecast_horizon=5,
            hidden_dim=128,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 256)
    
    def test_initialization(self, direction_head):
        """Test directional head initializes correctly"""
        assert direction_head.d_model == 256
        assert direction_head.forecast_horizon == 5
        assert direction_head.hidden_dim == 128
    
    def test_forward_pass(self, direction_head, sample_input):
        """Test forward pass produces valid probabilities"""
        output = direction_head(sample_input)
        
        assert output.shape == (2, 5)
        assert (output >= 0).all()  # Probabilities must be non-negative
        assert (output <= 1).all()  # Probabilities must be <= 1
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_probability_range(self, direction_head, sample_input):
        """Test that outputs are valid probabilities"""
        direction_head.eval()
        
        with torch.no_grad():
            output = direction_head(sample_input)
        
        # All outputs should be in [0, 1] range
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_gradient_flow(self, direction_head, sample_input):
        """Test gradients flow through directional head"""
        sample_input.requires_grad_(True)
        output = direction_head(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestResidualBlock:
    """Test suite for ResidualBlock"""
    
    @pytest.fixture
    def residual_block(self):
        """Create residual block instance"""
        return ResidualBlock(
            input_dim=128,
            hidden_dim=256,
            dropout=0.1,
            activation=nn.GELU()
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 128)
    
    def test_initialization(self, residual_block):
        """Test residual block initializes correctly"""
        assert isinstance(residual_block.linear1, nn.Linear)
        assert isinstance(residual_block.linear2, nn.Linear)
        assert isinstance(residual_block.activation, nn.GELU)
        assert isinstance(residual_block.dropout, nn.Dropout)
        assert isinstance(residual_block.layer_norm, nn.LayerNorm)
    
    def test_forward_pass(self, residual_block, sample_input):
        """Test forward pass with residual connection"""
        output = residual_block(sample_input)
        
        assert output.shape == sample_input.shape  # Same shape due to residual
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_residual_connection(self, sample_input):
        """Test that residual connection works correctly"""
        # Create block with identity-like initialization
        block = ResidualBlock(128, 128, dropout=0.0, activation=nn.Identity())
        
        # Initialize weights to near zero for testing residual effect
        with torch.no_grad():
            block.linear1.weight.fill_(0.01)
            block.linear1.bias.fill_(0)
            block.linear2.weight.fill_(0.01)
            block.linear2.bias.fill_(0)
        
        block.eval()
        
        with torch.no_grad():
            output = block(sample_input)
        
        # Output should be close to input due to residual connection
        # and small weights
        assert torch.allclose(output, sample_input, atol=0.1)
    
    def test_gradient_flow(self, residual_block, sample_input):
        """Test gradients flow through residual block"""
        sample_input.requires_grad_(True)
        output = residual_block(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestMultiTaskPredictionHead:
    """Test suite for MultiTaskPredictionHead"""
    
    @pytest.fixture
    def multi_task_head(self):
        """Create multi-task prediction head instance"""
        return MultiTaskPredictionHead(
            d_model=256,
            forecast_horizon=5,
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            hidden_dim=128,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 256)
    
    def test_initialization(self, multi_task_head):
        """Test multi-task head initializes correctly"""
        assert multi_task_head.d_model == 256
        assert multi_task_head.forecast_horizon == 5
        assert multi_task_head.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Check all task heads exist
        assert hasattr(multi_task_head, 'price_head')
        assert hasattr(multi_task_head, 'volatility_head')
        assert hasattr(multi_task_head, 'quantile_head')
        assert hasattr(multi_task_head, 'direction_head')
        assert hasattr(multi_task_head, 'shared_network')
    
    def test_forward_pass(self, multi_task_head, sample_input):
        """Test forward pass produces all task predictions"""
        output = multi_task_head(sample_input)
        
        assert isinstance(output, dict)
        assert set(output.keys()) == {"price", "volatility", "quantiles", "direction"}
        
        # Check shapes
        assert output["price"].shape == (2, 5)
        assert output["volatility"].shape == (2, 5)
        assert output["quantiles"].shape == (2, 5, 5)
        assert output["direction"].shape == (2, 5)
        
        # Check value constraints
        assert (output["volatility"] >= 0).all()  # Non-negative volatility
        assert (output["direction"] >= 0).all() and (output["direction"] <= 1).all()  # Valid probabilities
        
        # Check no NaN or Inf
        for task_output in output.values():
            assert not torch.isnan(task_output).any()
            assert not torch.isinf(task_output).any()
    
    def test_custom_task_weights(self):
        """Test with custom task weights"""
        custom_weights = {
            "price": 2.0,
            "volatility": 0.5,
            "quantiles": 1.5,
            "direction": 0.8
        }
        
        head = MultiTaskPredictionHead(task_weights=custom_weights)
        assert head.task_weights == custom_weights
    
    def test_3d_input_handling(self, multi_task_head):
        """Test handling of 3D input (sequence input)"""
        input_3d = torch.randn(2, 60, 256)
        output = multi_task_head(input_3d)
        
        # Should produce same output shapes as 2D input
        assert output["price"].shape == (2, 5)
        assert output["volatility"].shape == (2, 5)
        assert output["quantiles"].shape == (2, 5, 5)
        assert output["direction"].shape == (2, 5)
    
    def test_gradient_flow(self, multi_task_head, sample_input):
        """Test gradients flow through all task heads"""
        sample_input.requires_grad_(True)
        output = multi_task_head(sample_input)
        
        # Compute combined loss
        total_loss = (
            output["price"].sum() +
            output["volatility"].sum() +
            output["quantiles"].sum() +
            output["direction"].sum()
        )
        total_loss.backward()
        
        assert sample_input.grad is not None
        
        # Check gradients for all components
        for param in multi_task_head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestPredictionHeadFactory:
    """Test suite for prediction head factory function"""
    
    def test_create_price_head(self):
        """Test factory creates price head"""
        head = create_prediction_head("price", d_model=256, forecast_horizon=5)
        assert isinstance(head, PricePredictionHead)
        assert head.d_model == 256
        assert head.forecast_horizon == 5
    
    def test_create_volatility_head(self):
        """Test factory creates volatility head"""
        head = create_prediction_head("volatility", d_model=256, output_activation="relu")
        assert isinstance(head, VolatilityPredictionHead)
        assert isinstance(head.output_activation, nn.ReLU)
    
    def test_create_quantile_head(self):
        """Test factory creates quantile head"""
        quantiles = [0.1, 0.5, 0.9]
        head = create_prediction_head("quantiles", d_model=256, quantiles=quantiles)
        assert isinstance(head, QuantileRegressionHead)
        assert head.quantiles == quantiles
    
    def test_create_direction_head(self):
        """Test factory creates directional head"""
        head = create_prediction_head("direction", d_model=256, hidden_dim=64)
        assert isinstance(head, DirectionalPredictionHead)
        assert head.hidden_dim == 64
    
    def test_create_multi_task_head(self):
        """Test factory creates multi-task head"""
        head = create_prediction_head("multi_task", d_model=256, forecast_horizon=10)
        assert isinstance(head, MultiTaskPredictionHead)
        assert head.forecast_horizon == 10
    
    def test_invalid_head_type(self):
        """Test factory raises error for invalid head type"""
        with pytest.raises(ValueError):
            create_prediction_head("invalid_type")


class TestPredictionHeadPerformance:
    """Performance tests for prediction heads"""
    
    @pytest.mark.performance
    def test_prediction_speed(self):
        """Test prediction speed for different head types"""
        import time
        
        heads = {
            "price": PricePredictionHead(256, 5, dropout=0.0),
            "volatility": VolatilityPredictionHead(256, 5, dropout=0.0),
            "quantiles": QuantileRegressionHead(256, 5, dropout=0.0),
            "direction": DirectionalPredictionHead(256, 5, dropout=0.0),
            "multi_task": MultiTaskPredictionHead(256, 5, dropout=0.0),
        }
        
        input_tensor = torch.randn(32, 256)
        
        for name, head in heads.items():
            head.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = head(input_tensor)
            
            # Time prediction
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = head(input_tensor)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            
            # All heads should be fast
            assert avg_time < 0.01, f"{name} head too slow: {avg_time:.6f}s"
    
    def test_memory_efficiency(self):
        """Test memory usage of prediction heads"""
        import tracemalloc
        
        heads = [
            PricePredictionHead(256, 5),
            VolatilityPredictionHead(256, 5),
            QuantileRegressionHead(256, 5),
            DirectionalPredictionHead(256, 5),
            MultiTaskPredictionHead(256, 5),
        ]
        
        input_tensor = torch.randn(32, 256)
        
        for head in heads:
            tracemalloc.start()
            
            output = head(input_tensor)
            if isinstance(output, dict):
                loss = sum(v.sum() for v in output.values())
            else:
                loss = output.sum()
            loss.backward()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should be reasonable
            assert peak < 100 * 1024 * 1024  # 100MB max
    
    def test_batch_consistency(self):
        """Test batch processing consistency"""
        head = MultiTaskPredictionHead(256, 5)
        head.eval()
        
        # Single sample
        single_input = torch.randn(1, 256)
        
        # Batch with same sample repeated
        batch_input = single_input.repeat(5, 1)
        
        with torch.no_grad():
            single_output = head(single_input)
            batch_output = head(batch_input)
        
        # First item in batch should match single output for all tasks
        for task in single_output.keys():
            torch.testing.assert_close(
                single_output[task],
                batch_output[task][0:1],
                rtol=1e-5,
                atol=1e-5
            )