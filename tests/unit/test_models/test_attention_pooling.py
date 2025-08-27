"""
Unit tests for attention pooling components.
"""

import pytest
import torch
import torch.nn as nn
from src.models.components.attention_pooling import (
    AttentionPooling,
    MultiHeadAttentionPooling,
    HierarchicalAttentionPooling
)


class TestAttentionPooling:
    """Test suite for AttentionPooling"""
    
    @pytest.fixture
    def pooling_layer(self):
        """Create attention pooling layer for testing"""
        return AttentionPooling(d_model=256, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)  # (batch_size, seq_len, d_model)
    
    @pytest.fixture
    def sample_mask(self):
        """Generate sample mask tensor"""
        mask = torch.ones(4, 60)
        # Mask out last 10 positions for some samples
        mask[0, 50:] = 0
        mask[1, 55:] = 0
        return mask
    
    def test_initialization(self, pooling_layer):
        """Test attention pooling initializes correctly"""
        assert pooling_layer.d_model == 256
        assert pooling_layer.temperature == 1.0
        assert pooling_layer.query.shape == (256,)
        assert isinstance(pooling_layer.key_projection, nn.Linear)
        assert isinstance(pooling_layer.value_projection, nn.Linear)
    
    def test_forward_pass(self, pooling_layer, sample_input):
        """Test forward pass produces correct output shape"""
        output = pooling_layer(sample_input)
        
        assert output.shape == (4, 256)  # (batch_size, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_mask(self, pooling_layer, sample_input, sample_mask):
        """Test forward pass with mask"""
        output = pooling_layer(sample_input, mask=sample_mask)
        
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_return_attention_weights(self, pooling_layer, sample_input):
        """Test returning attention weights"""
        pooling_layer.eval()  # Set to eval mode to disable dropout
        output, attention_weights = pooling_layer(
            sample_input, return_attention=True
        )
        
        assert output.shape == (4, 256)
        assert attention_weights.shape == (4, 60)  # (batch_size, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(4), atol=1e-6)
        
        # Check attention weights are non-negative
        assert (attention_weights >= 0).all()
    
    def test_attention_weights_with_mask(self, pooling_layer, sample_input, sample_mask):
        """Test attention weights respect mask"""
        pooling_layer.eval()  # Set to eval mode to disable dropout
        output, attention_weights = pooling_layer(
            sample_input, mask=sample_mask, return_attention=True
        )
        
        # Check masked positions have zero attention
        assert torch.allclose(attention_weights[0, 50:], torch.zeros(10), atol=1e-6)
        assert torch.allclose(attention_weights[1, 55:], torch.zeros(5), atol=1e-6)
        
        # Check unmasked positions sum to 1
        assert torch.allclose(attention_weights[0, :50].sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(attention_weights[1, :55].sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_gradient_flow(self, pooling_layer, sample_input):
        """Test gradients flow through the layer"""
        sample_input.requires_grad_(True)
        output = pooling_layer(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()
    
    def test_temperature_scaling(self):
        """Test temperature scaling affects attention distribution"""
        d_model = 256
        sample_input = torch.randn(2, 10, d_model)
        
        # Use same pooling layer but modify temperature
        pooling = AttentionPooling(d_model, temperature=1.0)
        pooling.eval()  # Disable dropout for consistent results
        
        # Test with low temperature (sharper attention)
        pooling.temperature = 0.1
        _, attn_low = pooling(sample_input, return_attention=True)
        
        # Test with high temperature (smoother attention)
        pooling.temperature = 10.0
        _, attn_high = pooling(sample_input, return_attention=True)
        
        # Low temperature should have lower entropy (more focused)
        entropy_low = -(attn_low * torch.log(attn_low + 1e-8)).sum(dim=-1).mean()
        entropy_high = -(attn_high * torch.log(attn_high + 1e-8)).sum(dim=-1).mean()
        
        assert entropy_low < entropy_high


class TestMultiHeadAttentionPooling:
    """Test suite for MultiHeadAttentionPooling"""
    
    @pytest.fixture
    def pooling_layer(self):
        """Create multi-head attention pooling layer for testing"""
        return MultiHeadAttentionPooling(d_model=256, n_heads=8, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)
    
    def test_initialization(self, pooling_layer):
        """Test multi-head attention pooling initializes correctly"""
        assert pooling_layer.d_model == 256
        assert pooling_layer.n_heads == 8
        assert pooling_layer.d_k == 32  # 256 // 8
        assert pooling_layer.queries.shape == (8, 32)
    
    def test_forward_pass(self, pooling_layer, sample_input):
        """Test forward pass produces correct output shape"""
        output = pooling_layer(sample_input)
        
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_return_attention_weights(self, pooling_layer, sample_input):
        """Test returning multi-head attention weights"""
        output, attention_weights = pooling_layer(
            sample_input, return_attention=True
        )
        
        assert output.shape == (4, 256)
        assert attention_weights.shape == (4, 8, 60)  # (batch_size, n_heads, seq_len)
        
        # Check attention weights sum to 1 for each head
        assert torch.allclose(
            attention_weights.sum(dim=-1), 
            torch.ones(4, 8), 
            atol=1e-6
        )
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions"""
        with pytest.raises(AssertionError):
            MultiHeadAttentionPooling(d_model=255, n_heads=8)  # Not divisible
    
    def test_gradient_flow(self, pooling_layer, sample_input):
        """Test gradients flow through multi-head layer"""
        sample_input.requires_grad_(True)
        output = pooling_layer(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestHierarchicalAttentionPooling:
    """Test suite for HierarchicalAttentionPooling"""
    
    @pytest.fixture
    def pooling_layer(self):
        """Create hierarchical attention pooling layer for testing"""
        return HierarchicalAttentionPooling(d_model=256, window_size=10)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)
    
    def test_initialization(self, pooling_layer):
        """Test hierarchical attention pooling initializes correctly"""
        assert pooling_layer.d_model == 256
        assert pooling_layer.window_size == 10
        assert isinstance(pooling_layer.local_pooling, AttentionPooling)
        assert isinstance(pooling_layer.global_pooling, AttentionPooling)
    
    def test_forward_pass(self, pooling_layer, sample_input):
        """Test forward pass produces correct output shape"""
        output = pooling_layer(sample_input)
        
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_return_attention_weights(self, pooling_layer, sample_input):
        """Test returning hierarchical attention weights"""
        output, attention_weights = pooling_layer(
            sample_input, return_attention=True
        )
        
        assert output.shape == (4, 256)
        assert isinstance(attention_weights, tuple)
        assert len(attention_weights) == 2
        
        local_weights, global_weights = attention_weights
        assert local_weights.shape == (4, 6, 10)  # (batch_size, n_windows, window_size)
        assert global_weights.shape == (4, 6)  # (batch_size, n_windows)
    
    def test_padding_handling(self):
        """Test handling of sequences that don't divide evenly by window size"""
        pooling_layer = HierarchicalAttentionPooling(d_model=256, window_size=7)
        sample_input = torch.randn(2, 60, 256)  # 60 % 7 = 4, needs padding
        
        output = pooling_layer(sample_input)
        assert output.shape == (2, 256)
        assert not torch.isnan(output).any()
    
    def test_gradient_flow(self, pooling_layer, sample_input):
        """Test gradients flow through hierarchical layer"""
        sample_input.requires_grad_(True)
        output = pooling_layer(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestAttentionPoolingComparison:
    """Compare different attention pooling methods"""
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_output_consistency(self, sample_input):
        """Test all pooling methods produce valid outputs"""
        pooling_methods = [
            AttentionPooling(256),
            MultiHeadAttentionPooling(256, n_heads=8),
            HierarchicalAttentionPooling(256, window_size=10)
        ]
        
        outputs = []
        for pooling in pooling_methods:
            output = pooling(sample_input)
            outputs.append(output)
            
            assert output.shape == (2, 256)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        
        # Outputs should be different (different pooling strategies)
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-3)
    
    def test_attention_weight_properties(self, sample_input):
        """Test attention weight properties across methods"""
        pooling_methods = [
            AttentionPooling(256),
            MultiHeadAttentionPooling(256, n_heads=8),
            HierarchicalAttentionPooling(256, window_size=10)
        ]
        
        for pooling in pooling_methods:
            output, attention_weights = pooling(sample_input, return_attention=True)
            
            # Check output shape
            assert output.shape == (2, 256)
            
            # Check attention weights are valid probabilities
            if isinstance(attention_weights, tuple):
                # Hierarchical case
                local_weights, global_weights = attention_weights
                assert (local_weights >= 0).all()
                assert (global_weights >= 0).all()
                assert torch.allclose(local_weights.sum(dim=-1), torch.ones_like(local_weights.sum(dim=-1)), atol=1e-6)
                assert torch.allclose(global_weights.sum(dim=-1), torch.ones_like(global_weights.sum(dim=-1)), atol=1e-6)
            else:
                # Single or multi-head case
                assert (attention_weights >= 0).all()
                if attention_weights.dim() == 2:
                    # Single head
                    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2), atol=1e-6)
                else:
                    # Multi-head
                    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2, 8), atol=1e-6)