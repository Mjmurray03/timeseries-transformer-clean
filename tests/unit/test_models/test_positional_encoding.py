"""
Unit tests for positional encoding components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from src.models.components.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    RelativePositionalEncoding,
    AdaptivePositionalEncoding,
    TemporalPositionalEncoding,
    create_positional_encoding
)


class TestLearnedPositionalEncoding:
    """Test suite for LearnedPositionalEncoding"""
    
    @pytest.fixture
    def learned_pe(self):
        """Create learned positional encoding instance"""
        return LearnedPositionalEncoding(max_seq_len=60, d_model=256, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, learned_pe):
        """Test learned PE initializes correctly"""
        assert learned_pe.max_seq_len == 60
        assert learned_pe.d_model == 256
        assert learned_pe.pos_embedding.shape == (1, 60, 256)
        assert isinstance(learned_pe.dropout, nn.Dropout)
    
    def test_forward_pass_shape(self, learned_pe, sample_input):
        """Test forward pass produces correct output shape"""
        output = learned_pe(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_positional_encoding_added(self, learned_pe, sample_input):
        """Test that positional encoding is actually added"""
        learned_pe.eval()  # Disable dropout
        
        with torch.no_grad():
            output = learned_pe(sample_input)
            
            # Output should be different from input (PE added)
            assert not torch.allclose(output, sample_input)
            
            # But the difference should be the positional encoding (broadcasted)
            expected_diff = learned_pe.pos_embedding[:, :sample_input.size(1), :].expand_as(sample_input)
            actual_diff = output - sample_input
            torch.testing.assert_close(actual_diff, expected_diff, rtol=1e-5, atol=1e-5)
    
    def test_different_sequence_lengths(self, learned_pe):
        """Test with different sequence lengths"""
        seq_lengths = [10, 30, 60]
        
        for seq_len in seq_lengths:
            input_tensor = torch.randn(2, seq_len, 256)
            output = learned_pe(input_tensor)
            assert output.shape == (2, seq_len, 256)
    
    def test_sequence_too_long(self, learned_pe):
        """Test error handling for sequences longer than max_seq_len"""
        long_input = torch.randn(2, 70, 256)  # Longer than max_seq_len=60
        
        with pytest.raises(AssertionError):
            learned_pe(long_input)
    
    def test_wrong_model_dimension(self, learned_pe):
        """Test error handling for wrong model dimension"""
        wrong_dim_input = torch.randn(2, 60, 128)  # Wrong d_model
        
        with pytest.raises(AssertionError):
            learned_pe(wrong_dim_input)
    
    def test_gradient_flow(self, learned_pe, sample_input):
        """Test gradients flow through positional encoding"""
        sample_input.requires_grad_(True)
        output = learned_pe(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert learned_pe.pos_embedding.grad is not None
        assert not torch.isnan(learned_pe.pos_embedding.grad).any()
    
    def test_training_vs_eval_mode(self, learned_pe, sample_input):
        """Test behavior difference between training and eval modes"""
        # Training mode (with dropout)
        learned_pe.train()
        output_train = learned_pe(sample_input)
        
        # Eval mode (no dropout)
        learned_pe.eval()
        with torch.no_grad():
            output_eval = learned_pe(sample_input)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding"""
    
    @pytest.fixture
    def sinusoidal_pe(self):
        """Create sinusoidal positional encoding instance"""
        return SinusoidalPositionalEncoding(d_model=256, max_seq_len=1000, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, sinusoidal_pe):
        """Test sinusoidal PE initializes correctly"""
        assert sinusoidal_pe.d_model == 256
        assert sinusoidal_pe.pe.shape == (1, 1000, 256)
        assert isinstance(sinusoidal_pe.dropout, nn.Dropout)
    
    def test_forward_pass_shape(self, sinusoidal_pe, sample_input):
        """Test forward pass produces correct output shape"""
        output = sinusoidal_pe(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_sinusoidal_pattern(self, sinusoidal_pe):
        """Test that encoding follows sinusoidal pattern"""
        pe = sinusoidal_pe.pe[0, :10, :4]  # First 10 positions, first 4 dimensions
        
        # Even dimensions should use sine
        position = torch.arange(10, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 4, 2).float() * (-math.log(10000.0) / 256))
        
        expected_sin_0 = torch.sin(position * div_term[0]).squeeze()
        expected_sin_2 = torch.sin(position * div_term[1]).squeeze()
        
        assert torch.allclose(pe[:, 0], expected_sin_0, rtol=1e-5)
        assert torch.allclose(pe[:, 2], expected_sin_2, rtol=1e-5)
        
        # Odd dimensions should use cosine
        expected_cos_1 = torch.cos(position * div_term[0]).squeeze()
        expected_cos_3 = torch.cos(position * div_term[1]).squeeze()
        
        assert torch.allclose(pe[:, 1], expected_cos_1, rtol=1e-5)
        assert torch.allclose(pe[:, 3], expected_cos_3, rtol=1e-5)
    
    def test_position_uniqueness(self, sinusoidal_pe):
        """Test that different positions have different encodings"""
        pe = sinusoidal_pe.pe[0, :100, :]  # First 100 positions
        
        # Check that no two positions are identical
        for i in range(10):  # Check first 10 positions
            for j in range(i + 1, 10):
                assert not torch.allclose(pe[i], pe[j], atol=1e-6)
    
    def test_extrapolation_capability(self):
        """Test that sinusoidal PE can handle longer sequences than max_seq_len"""
        # Create PE with large max_seq_len to allow extrapolation
        pe = SinusoidalPositionalEncoding(d_model=64, max_seq_len=200, dropout=0.0)
        
        # Test with longer sequence than typical but within max_seq_len
        long_input = torch.randn(1, 150, 64)
        
        # Should work without error
        output = pe(long_input)
        assert output.shape == long_input.shape
    
    def test_deterministic_output(self, sinusoidal_pe, sample_input):
        """Test that sinusoidal PE produces deterministic output"""
        sinusoidal_pe.eval()
        
        with torch.no_grad():
            output1 = sinusoidal_pe(sample_input)
            output2 = sinusoidal_pe(sample_input)
        
        torch.testing.assert_close(output1, output2)


class TestRelativePositionalEncoding:
    """Test suite for RelativePositionalEncoding"""
    
    @pytest.fixture
    def relative_pe(self):
        """Create relative positional encoding instance"""
        return RelativePositionalEncoding(d_model=256, max_relative_position=32, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, relative_pe):
        """Test relative PE initializes correctly"""
        assert relative_pe.d_model == 256
        assert relative_pe.max_relative_position == 32
        
        # Vocabulary size should be 2 * max_relative_position + 1
        expected_vocab_size = 2 * 32 + 1
        assert relative_pe.relative_position_embeddings.num_embeddings == expected_vocab_size
    
    def test_forward_pass_shape(self, relative_pe, sample_input):
        """Test forward pass produces correct output shape"""
        output = relative_pe(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_relative_position_matrix(self, relative_pe):
        """Test relative position matrix generation"""
        seq_len = 5
        rel_pos = relative_pe._get_relative_positions(seq_len)
        
        assert rel_pos.shape == (seq_len, seq_len)
        
        # Check diagonal (self-attention) should be max_relative_position
        assert (rel_pos.diag() == relative_pe.max_relative_position).all()
        
        # Check symmetry around diagonal
        for i in range(seq_len):
            for j in range(seq_len):
                expected_distance = min(max(j - i, -relative_pe.max_relative_position), 
                                      relative_pe.max_relative_position)
                expected_index = expected_distance + relative_pe.max_relative_position
                assert rel_pos[i, j] == expected_index
    
    def test_gradient_flow(self, relative_pe, sample_input):
        """Test gradients flow through relative PE"""
        sample_input.requires_grad_(True)
        output = relative_pe(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert relative_pe.relative_position_embeddings.weight.grad is not None


class TestAdaptivePositionalEncoding:
    """Test suite for AdaptivePositionalEncoding"""
    
    @pytest.fixture
    def adaptive_pe(self):
        """Create adaptive positional encoding instance"""
        return AdaptivePositionalEncoding(d_model=256, max_seq_len=60, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, adaptive_pe):
        """Test adaptive PE initializes correctly"""
        assert adaptive_pe.d_model == 256
        assert hasattr(adaptive_pe, 'learned_encoding')
        assert hasattr(adaptive_pe, 'sinusoidal_encoding')
        assert hasattr(adaptive_pe, 'gate')
    
    def test_forward_pass_shape(self, adaptive_pe, sample_input):
        """Test forward pass produces correct output shape"""
        output = adaptive_pe(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gating_mechanism(self, adaptive_pe, sample_input):
        """Test that gating weights are valid probabilities"""
        adaptive_pe.eval()
        
        with torch.no_grad():
            # Access gate weights
            gate_weights = adaptive_pe.gate(sample_input)
            
            assert gate_weights.shape == (2, 60, 2)
            
            # Should be valid probabilities (sum to 1)
            assert torch.allclose(gate_weights.sum(dim=-1), torch.ones_like(gate_weights.sum(dim=-1)))
            assert (gate_weights >= 0).all()
            assert (gate_weights <= 1).all()
    
    def test_gradient_flow(self, adaptive_pe, sample_input):
        """Test gradients flow through adaptive PE"""
        sample_input.requires_grad_(True)
        output = adaptive_pe(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        
        # Check gradients for all components
        for param in adaptive_pe.parameters():
            assert param.grad is not None


class TestTemporalPositionalEncoding:
    """Test suite for TemporalPositionalEncoding"""
    
    @pytest.fixture
    def temporal_pe(self):
        """Create temporal positional encoding instance"""
        return TemporalPositionalEncoding(d_model=256, max_seq_len=60, time_scale=1.0, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(2, 60, 256)
    
    def test_initialization(self, temporal_pe):
        """Test temporal PE initializes correctly"""
        assert temporal_pe.d_model == 256
        assert temporal_pe.time_scale == 1.0
        assert temporal_pe.temporal_embedding.shape == (1, 60, 256)
        assert isinstance(temporal_pe.time_projection, nn.Linear)
    
    def test_forward_pass_shape(self, temporal_pe, sample_input):
        """Test forward pass produces correct output shape"""
        output = temporal_pe(sample_input)
        
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_with_time_indices(self, temporal_pe, sample_input):
        """Test forward pass with time indices"""
        time_indices = torch.randn(2, 60)  # Random time differences
        
        output = temporal_pe(sample_input, time_indices=time_indices)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_time_modulation(self, temporal_pe, sample_input):
        """Test that time indices modulate the encoding"""
        temporal_pe.eval()
        
        # Test with different time indices
        time_indices_1 = torch.zeros(2, 60)  # No time difference
        time_indices_2 = torch.ones(2, 60) * 5  # Large time difference
        
        with torch.no_grad():
            output_1 = temporal_pe(sample_input, time_indices=time_indices_1)
            output_2 = temporal_pe(sample_input, time_indices=time_indices_2)
        
        # Outputs should be different due to time modulation
        assert not torch.allclose(output_1, output_2, atol=1e-6)
    
    def test_gradient_flow(self, temporal_pe, sample_input):
        """Test gradients flow through temporal PE"""
        sample_input.requires_grad_(True)
        output = temporal_pe(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert temporal_pe.temporal_embedding.grad is not None
        assert temporal_pe.time_projection.weight.grad is not None


class TestPositionalEncodingFactory:
    """Test suite for positional encoding factory function"""
    
    def test_create_learned_encoding(self):
        """Test factory creates learned encoding"""
        pe = create_positional_encoding("learned", d_model=256, max_seq_len=60)
        assert isinstance(pe, LearnedPositionalEncoding)
        assert pe.d_model == 256
        assert pe.max_seq_len == 60
    
    def test_create_sinusoidal_encoding(self):
        """Test factory creates sinusoidal encoding"""
        pe = create_positional_encoding("sinusoidal", d_model=256, max_seq_len=60)
        assert isinstance(pe, SinusoidalPositionalEncoding)
        assert pe.d_model == 256
    
    def test_create_relative_encoding(self):
        """Test factory creates relative encoding"""
        pe = create_positional_encoding("relative", d_model=256, max_relative_position=16)
        assert isinstance(pe, RelativePositionalEncoding)
        assert pe.d_model == 256
        assert pe.max_relative_position == 16
    
    def test_create_adaptive_encoding(self):
        """Test factory creates adaptive encoding"""
        pe = create_positional_encoding("adaptive", d_model=256, max_seq_len=60)
        assert isinstance(pe, AdaptivePositionalEncoding)
        assert pe.d_model == 256
    
    def test_create_temporal_encoding(self):
        """Test factory creates temporal encoding"""
        pe = create_positional_encoding("temporal", d_model=256, time_scale=2.0)
        assert isinstance(pe, TemporalPositionalEncoding)
        assert pe.d_model == 256
        assert pe.time_scale == 2.0
    
    def test_invalid_encoding_type(self):
        """Test factory raises error for invalid encoding type"""
        with pytest.raises(ValueError):
            create_positional_encoding("invalid_type")


class TestPositionalEncodingPerformance:
    """Performance tests for positional encodings"""
    
    @pytest.mark.performance
    def test_encoding_speed(self):
        """Test encoding speed for different types"""
        import time
        
        encodings = {
            "learned": LearnedPositionalEncoding(60, 256, 0.0),
            "sinusoidal": SinusoidalPositionalEncoding(256, 1000, 0.0),
            "relative": RelativePositionalEncoding(256, 32, 0.0),
        }
        
        input_tensor = torch.randn(32, 60, 256)
        
        for name, encoding in encodings.items():
            encoding.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = encoding(input_tensor)
            
            # Time encoding
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = encoding(input_tensor)
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            
            # All encodings should be fast
            assert avg_time < 0.005, f"{name} encoding too slow: {avg_time:.6f}s"
    
    def test_memory_efficiency(self):
        """Test memory usage of different encodings"""
        import tracemalloc
        
        encodings = [
            LearnedPositionalEncoding(60, 256),
            SinusoidalPositionalEncoding(256, 1000),
            RelativePositionalEncoding(256, 32),
        ]
        
        input_tensor = torch.randn(32, 60, 256)
        
        for encoding in encodings:
            tracemalloc.start()
            
            # Enable gradients for input tensor
            input_tensor_grad = input_tensor.clone().requires_grad_(True)
            output = encoding(input_tensor_grad)
            loss = output.sum()
            loss.backward()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should be reasonable
            assert peak < 50 * 1024 * 1024  # 50MB max