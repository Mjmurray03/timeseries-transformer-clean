"""
Unit tests for transformer block components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.components.transformer_block import (
    TransformerBlock,
    InterpretableTransformerBlock,
    create_causal_mask,
    create_padding_mask
)


class TestTransformerBlock:
    """Test suite for TransformerBlock"""
    
    @pytest.fixture
    def transformer_block(self):
        """Create transformer block instance for testing"""
        return TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)  # (batch_size, seq_len, d_model)
    
    def test_initialization(self, transformer_block):
        """Test transformer block initializes correctly"""
        assert transformer_block is not None
        assert transformer_block.d_model == 256
        assert transformer_block.n_heads == 8
        assert transformer_block.d_ff == 1024
        assert transformer_block.dropout == 0.1
        
        # Check components exist
        assert hasattr(transformer_block, 'attention')
        assert hasattr(transformer_block, 'norm1')
        assert hasattr(transformer_block, 'ffn')
        assert hasattr(transformer_block, 'norm2')
    
    def test_forward_pass_shape(self, transformer_block, sample_input):
        """Test forward pass produces correct output shape"""
        output = transformer_block(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_attention_weights(self, transformer_block, sample_input):
        """Test forward pass with attention weight return"""
        output, attention_weights = transformer_block(sample_input, return_attention=True)
        
        assert output.shape == sample_input.shape
        assert attention_weights is not None
        assert attention_weights.shape == (2, 60, 60)  # (batch, seq, seq) - PyTorch averages across heads
        
        # Check attention weights are reasonable (dropout can affect exact sum)
        attention_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=0.1)  # Allow for dropout
        assert (attention_weights >= 0).all()
        assert (attention_weights <= 1).all()
    
    def test_gradient_flow(self, transformer_block, sample_input):
        """Test gradients flow through the block"""
        sample_input.requires_grad_(True)
        output = transformer_block(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()
        
        # Check model parameters have gradients
        for param in transformer_block.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_different_batch_sizes(self, transformer_block):
        """Test block works with different batch sizes"""
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 60, 256)
            output = transformer_block(input_tensor)
            assert output.shape == (batch_size, 60, 256)
    
    def test_different_sequence_lengths(self, transformer_block):
        """Test block works with different sequence lengths"""
        seq_lengths = [10, 30, 60, 120]
        
        for seq_len in seq_lengths:
            input_tensor = torch.randn(2, seq_len, 256)
            output = transformer_block(input_tensor)
            assert output.shape == (2, seq_len, 256)
    
    def test_with_mask(self, transformer_block, sample_input):
        """Test forward pass with attention mask"""
        seq_len = sample_input.size(1)
        mask = create_causal_mask(seq_len, sample_input.device)
        
        output = transformer_block(sample_input, mask=mask)
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
    
    def test_training_vs_eval_mode(self, transformer_block, sample_input):
        """Test behavior difference between training and eval modes"""
        # Training mode
        transformer_block.train()
        output_train = transformer_block(sample_input)
        
        # Eval mode
        transformer_block.eval()
        with torch.no_grad():
            output_eval = transformer_block(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_deterministic_output(self, transformer_block, sample_input):
        """Test model produces consistent outputs in eval mode"""
        transformer_block.eval()
        
        with torch.no_grad():
            output1 = transformer_block(sample_input)
            output2 = transformer_block(sample_input)
        
        torch.testing.assert_close(output1, output2)
    
    def test_parameter_count(self, transformer_block):
        """Test parameter count is reasonable"""
        total_params = sum(p.numel() for p in transformer_block.parameters())
        
        # Rough estimate: attention (4 * 256^2) + ffn (2 * 256 * 1024) + norms
        expected_params = 4 * 256 * 256 + 2 * 256 * 1024 + 2 * 256
        
        # Allow some tolerance for bias terms
        assert abs(total_params - expected_params) < 10000
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions"""
        with pytest.raises(AssertionError):
            # d_model not divisible by n_heads
            TransformerBlock(d_model=255, n_heads=8)


class TestInterpretableTransformerBlock:
    """Test suite for InterpretableTransformerBlock"""
    
    @pytest.fixture
    def interpretable_block(self):
        """Create interpretable transformer block for testing"""
        return InterpretableTransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, interpretable_block):
        """Test interpretable block initializes correctly"""
        assert interpretable_block is not None
        assert interpretable_block.d_model == 256
        assert interpretable_block.n_heads == 8
        assert interpretable_block.d_k == 32  # 256 / 8
        
        # Check separate Q, K, V projections
        assert hasattr(interpretable_block, 'W_q')
        assert hasattr(interpretable_block, 'W_k')
        assert hasattr(interpretable_block, 'W_v')
        assert hasattr(interpretable_block, 'W_o')
    
    def test_forward_pass(self, interpretable_block, sample_input):
        """Test forward pass produces correct output"""
        output = interpretable_block(sample_input)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_weights_interpretability(self, interpretable_block, sample_input):
        """Test attention weights for interpretability"""
        output, attention_weights = interpretable_block(sample_input, return_attention=True)
        
        assert output.shape == sample_input.shape
        assert attention_weights.shape == (2, 60, 60)  # Averaged across heads
        
        # Check attention weights are reasonable (averaging and dropout can affect exact sum)
        attention_sums = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=0.1)  # Allow for dropout and averaging
        assert (attention_weights >= 0).all()
        assert (attention_weights <= 1).all()
    
    def test_gradient_flow(self, interpretable_block, sample_input):
        """Test gradients flow through interpretable block"""
        sample_input.requires_grad_(True)
        output = interpretable_block(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        
        # Check all Q, K, V, O projections have gradients
        for name, param in interpretable_block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestMaskingFunctions:
    """Test suite for masking utility functions"""
    
    def test_create_causal_mask(self):
        """Test causal mask creation"""
        seq_len = 5
        device = torch.device('cpu')
        mask = create_causal_mask(seq_len, device)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check causal structure (lower triangular)
        expected = torch.tensor([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True]
        ])
        
        torch.testing.assert_close(mask, expected)
    
    def test_create_padding_mask(self):
        """Test padding mask creation"""
        lengths = torch.tensor([3, 5, 2])
        max_len = 5
        mask = create_padding_mask(lengths, max_len)
        
        assert mask.shape == (3, 5)
        assert mask.dtype == torch.bool
        
        expected = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False]
        ])
        
        torch.testing.assert_close(mask, expected)
    
    def test_empty_sequences(self):
        """Test mask creation with empty sequences"""
        lengths = torch.tensor([0, 2, 0])
        max_len = 3
        mask = create_padding_mask(lengths, max_len)
        
        expected = torch.tensor([
            [False, False, False],
            [True, True, False],
            [False, False, False]
        ])
        
        torch.testing.assert_close(mask, expected)


class TestTransformerBlockPerformance:
    """Performance and memory tests for transformer blocks"""
    
    @pytest.mark.performance
    def test_inference_speed(self):
        """Test inference speed meets requirements"""
        import time
        
        block = TransformerBlock(d_model=256, n_heads=8)
        block.eval()
        
        input_tensor = torch.randn(32, 60, 256)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = block(input_tensor)
        
        # Time inference
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = block(input_tensor)
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        p99_time = np.percentile(times, 99)
        
        # Should be fast for single block (more realistic for CPU)
        assert avg_time < 0.05  # 50ms average
        assert p99_time < 0.1  # 100ms P99
    
    @pytest.mark.memory
    def test_memory_usage(self):
        """Test memory usage is reasonable"""
        import tracemalloc
        
        tracemalloc.start()
        
        block = TransformerBlock(d_model=256, n_heads=8)
        input_tensor = torch.randn(32, 60, 256)
        
        # Forward pass
        output = block(input_tensor)
        loss = output.sum()
        loss.backward()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (< 100MB for single block)
        assert peak < 100 * 1024 * 1024  # 100MB
    
    def test_batch_consistency(self):
        """Test batch processing consistency"""
        block = TransformerBlock(d_model=256, n_heads=8)
        block.eval()
        
        # Single sample
        single_input = torch.randn(1, 60, 256)
        
        # Batch with same sample repeated
        batch_input = single_input.repeat(5, 1, 1)
        
        with torch.no_grad():
            single_output = block(single_input)
            batch_output = block(batch_input)
        
        # First item in batch should match single output
        torch.testing.assert_close(
            single_output,
            batch_output[0:1],
            rtol=1e-5,
            atol=1e-5
        )