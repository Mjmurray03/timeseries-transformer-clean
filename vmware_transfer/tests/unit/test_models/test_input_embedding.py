"""
Unit tests for input embedding components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.components.input_embedding import (
    InputEmbedding,
    FeatureWiseEmbedding,
    ScaledInputEmbedding,
    AdaptiveInputEmbedding,
    MultiScaleEmbedding,
    create_input_embedding
)


class TestInputEmbedding:
    """Test suite for InputEmbedding"""
    
    @pytest.fixture
    def input_embedding(self):
        """Create input embedding instance for testing"""
        return InputEmbedding(input_dim=7, d_model=256, dropout=0.1, use_layer_norm=True)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 7)  # (batch_size, seq_len, input_dim)
    
    def test_initialization(self, input_embedding):
        """Test input embedding initializes correctly"""
        assert input_embedding.input_dim == 7
        assert input_embedding.d_model == 256
        assert input_embedding.dropout == 0.1
        assert input_embedding.use_layer_norm == True
        
        # Check components exist
        assert isinstance(input_embedding.projection, nn.Linear)
        assert isinstance(input_embedding.layer_norm, nn.LayerNorm)
        assert isinstance(input_embedding.dropout_layer, nn.Dropout)
    
    def test_forward_pass_shape(self, input_embedding, sample_input):
        """Test forward pass produces correct output shape"""
        output = input_embedding(sample_input)
        
        assert output.shape == (2, 60, 256)  # (batch_size, seq_len, d_model)
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_without_layer_norm(self):
        """Test embedding without layer normalization"""
        embedding = InputEmbedding(input_dim=7, d_model=256, use_layer_norm=False)
        sample_input = torch.randn(2, 60, 7)
        
        output = embedding(sample_input)
        assert output.shape == (2, 60, 256)
        assert not hasattr(embedding, 'layer_norm') or embedding.layer_norm is None
    
    def test_different_dimensions(self):
        """Test with different input and model dimensions"""
        embedding = InputEmbedding(input_dim=10, d_model=128)
        sample_input = torch.randn(4, 30, 10)
        
        output = embedding(sample_input)
        assert output.shape == (4, 30, 128)
    
    def test_wrong_input_dimension(self, input_embedding):
        """Test error handling for wrong input dimension"""
        wrong_input = torch.randn(2, 60, 5)  # Wrong input_dim
        
        with pytest.raises(AssertionError):
            input_embedding(wrong_input)
    
    def test_gradient_flow(self, input_embedding, sample_input):
        """Test gradients flow through embedding"""
        sample_input.requires_grad_(True)
        output = input_embedding(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert input_embedding.projection.weight.grad is not None
        assert not torch.isnan(input_embedding.projection.weight.grad).any()
    
    def test_training_vs_eval_mode(self, input_embedding, sample_input):
        """Test behavior difference between training and eval modes"""
        # Training mode
        input_embedding.train()
        output_train = input_embedding(sample_input)
        
        # Eval mode
        input_embedding.eval()
        with torch.no_grad():
            output_eval = input_embedding(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_deterministic_output(self, input_embedding, sample_input):
        """Test model produces consistent outputs in eval mode"""
        input_embedding.eval()
        
        with torch.no_grad():
            output1 = input_embedding(sample_input)
            output2 = input_embedding(sample_input)
        
        torch.testing.assert_close(output1, output2)


class TestFeatureWiseEmbedding:
    """Test suite for FeatureWiseEmbedding"""
    
    @pytest.fixture
    def feature_names(self):
        """Standard financial feature names"""
        return ["open", "high", "low", "close", "volume", "returns", "volume_ratio"]
    
    @pytest.fixture
    def feature_embedding(self, feature_names):
        """Create feature-wise embedding instance"""
        return FeatureWiseEmbedding(
            input_dim=7, 
            d_model=256, 
            feature_names=feature_names, 
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 7)
    
    def test_initialization(self, feature_embedding, feature_names):
        """Test feature-wise embedding initializes correctly"""
        assert feature_embedding.input_dim == 7
        assert feature_embedding.d_model == 256
        assert feature_embedding.feature_names == feature_names
        
        # Check feature embeddings exist
        assert len(feature_embedding.feature_embeddings) == 7
        for name in feature_names:
            assert name in feature_embedding.feature_embeddings
            assert isinstance(feature_embedding.feature_embeddings[name], nn.Linear)
    
    def test_forward_pass_shape(self, feature_embedding, sample_input):
        """Test forward pass produces correct output shape"""
        output = feature_embedding(sample_input)
        
        assert output.shape == (2, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_default_feature_names(self):
        """Test with default feature names"""
        embedding = FeatureWiseEmbedding(input_dim=5, d_model=128)
        
        expected_names = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        assert embedding.feature_names == expected_names
    
    def test_mismatched_feature_names(self):
        """Test error handling for mismatched feature names"""
        with pytest.raises(AssertionError):
            FeatureWiseEmbedding(input_dim=7, feature_names=["a", "b", "c"])  # Only 3 names for 7 features
    
    def test_gradient_flow(self, feature_embedding, sample_input):
        """Test gradients flow through all feature embeddings"""
        sample_input.requires_grad_(True)
        output = feature_embedding(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        
        # Check gradients for all feature embeddings
        for name, embedding in feature_embedding.feature_embeddings.items():
            assert embedding.weight.grad is not None, f"No gradient for feature {name}"
            assert not torch.isnan(embedding.weight.grad).any()


class TestScaledInputEmbedding:
    """Test suite for ScaledInputEmbedding"""
    
    @pytest.fixture
    def scaled_embedding(self):
        """Create scaled input embedding instance"""
        return ScaledInputEmbedding(input_dim=7, d_model=256, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 7)
    
    def test_initialization(self, scaled_embedding):
        """Test scaled embedding initializes correctly"""
        assert scaled_embedding.input_dim == 7
        assert scaled_embedding.d_model == 256
        
        # Check scale parameter
        assert isinstance(scaled_embedding.scale, nn.Parameter)
        expected_scale = np.sqrt(256)
        assert abs(scaled_embedding.scale.item() - expected_scale) < 1e-6
    
    def test_custom_scale_factor(self):
        """Test with custom scale factor"""
        custom_scale = 10.0
        embedding = ScaledInputEmbedding(input_dim=7, d_model=256, scale_factor=custom_scale)
        
        assert abs(embedding.scale.item() - custom_scale) < 1e-6
    
    def test_forward_pass_shape(self, scaled_embedding, sample_input):
        """Test forward pass produces correct output shape"""
        output = scaled_embedding(sample_input)
        
        assert output.shape == (2, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_scaling_effect(self, sample_input):
        """Test that scaling actually affects the output"""
        embedding1 = ScaledInputEmbedding(input_dim=7, d_model=256, scale_factor=1.0, dropout=0.0)
        embedding2 = ScaledInputEmbedding(input_dim=7, d_model=256, scale_factor=2.0, dropout=0.0)
        
        # Copy weights to make embeddings identical except for scale
        embedding2.projection.weight.data = embedding1.projection.weight.data.clone()
        embedding2.projection.bias.data = embedding1.projection.bias.data.clone()
        embedding2.layer_norm.weight.data = embedding1.layer_norm.weight.data.clone()
        embedding2.layer_norm.bias.data = embedding1.layer_norm.bias.data.clone()
        
        embedding1.eval()
        embedding2.eval()
        
        with torch.no_grad():
            output1 = embedding1(sample_input)
            output2 = embedding2(sample_input)
        
        # Outputs should be different due to different scaling
        assert not torch.allclose(output1, output2, atol=1e-6)
    
    def test_learnable_scale(self, scaled_embedding, sample_input):
        """Test that scale parameter is learnable"""
        initial_scale = scaled_embedding.scale.item()
        
        output = scaled_embedding(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert scaled_embedding.scale.grad is not None
        assert not torch.isnan(scaled_embedding.scale.grad).any()


class TestAdaptiveInputEmbedding:
    """Test suite for AdaptiveInputEmbedding"""
    
    @pytest.fixture
    def adaptive_embedding(self):
        """Create adaptive input embedding instance"""
        return AdaptiveInputEmbedding(input_dim=7, d_model=256, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 7)
    
    def test_initialization(self, adaptive_embedding):
        """Test adaptive embedding initializes correctly"""
        assert adaptive_embedding.input_dim == 7
        assert adaptive_embedding.d_model == 256
        
        # Check running statistics buffers
        assert hasattr(adaptive_embedding, 'running_mean')
        assert hasattr(adaptive_embedding, 'running_var')
        assert hasattr(adaptive_embedding, 'num_batches_tracked')
        
        assert adaptive_embedding.running_mean.shape == (7,)
        assert adaptive_embedding.running_var.shape == (7,)
        
        # Check learnable parameters
        assert adaptive_embedding.weight.shape == (7,)
        assert adaptive_embedding.bias.shape == (7,)
    
    def test_forward_pass_shape(self, adaptive_embedding, sample_input):
        """Test forward pass produces correct output shape"""
        output = adaptive_embedding(sample_input)
        
        assert output.shape == (2, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_running_statistics_update(self, adaptive_embedding, sample_input):
        """Test that running statistics are updated during training"""
        adaptive_embedding.train()
        
        initial_mean = adaptive_embedding.running_mean.clone()
        initial_var = adaptive_embedding.running_var.clone()
        initial_count = adaptive_embedding.num_batches_tracked.item()
        
        # Forward pass should update statistics
        _ = adaptive_embedding(sample_input)
        
        # Statistics should have changed
        assert not torch.allclose(adaptive_embedding.running_mean, initial_mean)
        assert not torch.allclose(adaptive_embedding.running_var, initial_var)
        assert adaptive_embedding.num_batches_tracked.item() == initial_count + 1
    
    def test_no_statistics_update_in_eval(self, adaptive_embedding, sample_input):
        """Test that running statistics are not updated in eval mode"""
        adaptive_embedding.eval()
        
        initial_mean = adaptive_embedding.running_mean.clone()
        initial_var = adaptive_embedding.running_var.clone()
        initial_count = adaptive_embedding.num_batches_tracked.item()
        
        with torch.no_grad():
            _ = adaptive_embedding(sample_input)
        
        # Statistics should not have changed
        torch.testing.assert_close(adaptive_embedding.running_mean, initial_mean)
        torch.testing.assert_close(adaptive_embedding.running_var, initial_var)
        assert adaptive_embedding.num_batches_tracked.item() == initial_count
    
    def test_gradient_flow(self, adaptive_embedding, sample_input):
        """Test gradients flow through adaptive parameters"""
        sample_input.requires_grad_(True)
        output = adaptive_embedding(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert adaptive_embedding.weight.grad is not None
        assert adaptive_embedding.bias.grad is not None
        assert adaptive_embedding.projection.weight.grad is not None


class TestMultiScaleEmbedding:
    """Test suite for MultiScaleEmbedding"""
    
    @pytest.fixture
    def multi_scale_embedding(self):
        """Create multi-scale embedding instance"""
        return MultiScaleEmbedding(
            input_dim=7, 
            d_model=256, 
            kernel_sizes=[1, 3, 5, 7], 
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 7)
    
    def test_initialization(self, multi_scale_embedding):
        """Test multi-scale embedding initializes correctly"""
        assert multi_scale_embedding.input_dim == 7
        assert multi_scale_embedding.d_model == 256
        assert multi_scale_embedding.kernel_sizes == [1, 3, 5, 7]
        
        # Check convolutional layers
        assert len(multi_scale_embedding.conv_layers) == 4
        for i, conv in enumerate(multi_scale_embedding.conv_layers):
            assert isinstance(conv, nn.Conv1d)
            assert conv.in_channels == 7
            assert conv.out_channels == 256 // 4  # d_model divided by number of scales
            assert conv.kernel_size == (multi_scale_embedding.kernel_sizes[i],)
    
    def test_forward_pass_shape(self, multi_scale_embedding, sample_input):
        """Test forward pass produces correct output shape"""
        output = multi_scale_embedding(sample_input)
        
        assert output.shape == (2, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes"""
        kernel_sizes = [1, 5, 9]
        embedding = MultiScaleEmbedding(
            input_dim=7, 
            d_model=192,  # Divisible by 3
            kernel_sizes=kernel_sizes
        )
        
        sample_input = torch.randn(2, 60, 7)
        output = embedding(sample_input)
        
        assert output.shape == (2, 60, 192)
        assert len(embedding.conv_layers) == 3
    
    def test_dimension_adjustment(self):
        """Test final projection when dimensions don't divide evenly"""
        # d_model=256 doesn't divide evenly by 3 kernel sizes
        embedding = MultiScaleEmbedding(
            input_dim=7, 
            d_model=256, 
            kernel_sizes=[1, 3, 5]  # 3 kernels
        )
        
        # Should have final projection layer
        assert isinstance(embedding.final_projection, nn.Linear)
        
        sample_input = torch.randn(2, 60, 7)
        output = embedding(sample_input)
        assert output.shape == (2, 60, 256)
    
    def test_gradient_flow(self, multi_scale_embedding, sample_input):
        """Test gradients flow through all convolutional layers"""
        sample_input.requires_grad_(True)
        output = multi_scale_embedding(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        
        # Check gradients for all conv layers
        for i, conv in enumerate(multi_scale_embedding.conv_layers):
            assert conv.weight.grad is not None, f"No gradient for conv layer {i}"
            assert not torch.isnan(conv.weight.grad).any()


class TestInputEmbeddingFactory:
    """Test suite for input embedding factory function"""
    
    def test_create_basic_embedding(self):
        """Test factory creates basic embedding"""
        embedding = create_input_embedding("basic", input_dim=7, d_model=256)
        assert isinstance(embedding, InputEmbedding)
        assert embedding.input_dim == 7
        assert embedding.d_model == 256
    
    def test_create_feature_wise_embedding(self):
        """Test factory creates feature-wise embedding"""
        feature_names = ["a", "b", "c", "d", "e", "f", "g"]
        embedding = create_input_embedding(
            "feature_wise", 
            input_dim=7, 
            d_model=256, 
            feature_names=feature_names
        )
        assert isinstance(embedding, FeatureWiseEmbedding)
        assert embedding.feature_names == feature_names
    
    def test_create_scaled_embedding(self):
        """Test factory creates scaled embedding"""
        embedding = create_input_embedding("scaled", input_dim=7, d_model=256, scale_factor=5.0)
        assert isinstance(embedding, ScaledInputEmbedding)
        assert abs(embedding.scale.item() - 5.0) < 1e-6
    
    def test_create_adaptive_embedding(self):
        """Test factory creates adaptive embedding"""
        embedding = create_input_embedding("adaptive", input_dim=7, d_model=256, momentum=0.2)
        assert isinstance(embedding, AdaptiveInputEmbedding)
        assert embedding.momentum == 0.2
    
    def test_create_multi_scale_embedding(self):
        """Test factory creates multi-scale embedding"""
        kernel_sizes = [1, 3, 7]
        embedding = create_input_embedding(
            "multi_scale", 
            input_dim=7, 
            d_model=256, 
            kernel_sizes=kernel_sizes
        )
        assert isinstance(embedding, MultiScaleEmbedding)
        assert embedding.kernel_sizes == kernel_sizes
    
    def test_invalid_embedding_type(self):
        """Test factory raises error for invalid embedding type"""
        with pytest.raises(ValueError):
            create_input_embedding("invalid_type")


class TestInputEmbeddingPerformance:
    """Performance tests for input embeddings"""
    
    @pytest.mark.performance
    def test_embedding_speed(self):
        """Test embedding speed for different types"""
        import time
        
        embeddings = {
            "basic": InputEmbedding(7, 256, 0.0),
            "feature_wise": FeatureWiseEmbedding(7, 256, dropout=0.0),
            "scaled": ScaledInputEmbedding(7, 256, dropout=0.0),
            "multi_scale": MultiScaleEmbedding(7, 256, dropout=0.0),
        }
        
        input_tensor = torch.randn(32, 60, 7)
        
        for name, embedding in embeddings.items():
            embedding.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = embedding(input_tensor)
            
            # Time embedding
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = embedding(input_tensor)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            
            # All embeddings should be fast
            assert avg_time < 0.01, f"{name} embedding too slow: {avg_time:.6f}s"
    
    def test_memory_efficiency(self):
        """Test memory usage of different embeddings"""
        import tracemalloc
        
        embeddings = [
            InputEmbedding(7, 256),
            FeatureWiseEmbedding(7, 256),
            ScaledInputEmbedding(7, 256),
            MultiScaleEmbedding(7, 256),
        ]
        
        input_tensor = torch.randn(32, 60, 7)
        
        for embedding in embeddings:
            tracemalloc.start()
            
            output = embedding(input_tensor)
            loss = output.sum()
            loss.backward()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should be reasonable
            assert peak < 100 * 1024 * 1024  # 100MB max
    
    def test_batch_consistency(self):
        """Test batch processing consistency"""
        embedding = InputEmbedding(7, 256)
        embedding.eval()
        
        # Single sample
        single_input = torch.randn(1, 60, 7)
        
        # Batch with same sample repeated
        batch_input = single_input.repeat(5, 1, 1)
        
        with torch.no_grad():
            single_output = embedding(single_input)
            batch_output = embedding(batch_input)
        
        # First item in batch should match single output
        torch.testing.assert_close(
            single_output,
            batch_output[0:1],
            rtol=1e-5,
            atol=1e-5
        )