"""
Comprehensive unit tests for model components.
Follows patterns from testing-standards.md with 85% coverage requirement for model components.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from unittest.mock import Mock, patch
from torch.testing import assert_close

from src.models.components.transformer_block import TransformerBlock
from src.models.components.attention_pooling import AttentionPooling
from src.models.components.positional_encoding import PositionalEncoding
from src.models.components.prediction_heads import PredictionHeads
from src.models.components.input_embedding import InputEmbedding
from src.models.components.interpretable_attention import InterpretableAttention
from src.models.components.temporal_masking import TemporalMasking
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.models.losses.composite_loss import CompositeLoss
from src.models.losses.quantile_loss import QuantileLoss
from src.models.losses.directional_loss import DirectionalLoss


class TestTransformerBlock:
    """Test suite for TransformerBlock following testing-standards.md patterns"""
    
    @pytest.fixture
    def transformer_block(self):
        """Create TransformerBlock instance for testing"""
        return TransformerBlock(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
    
    @pytest.fixture
    def input_tensor(self):
        """Generate input tensor for transformer testing"""
        torch.manual_seed(42)
        return torch.randn(16, 60, 256)  # batch_size, seq_len, d_model
    
    def test_initialization(self, transformer_block):
        """Test transformer block initializes correctly"""
        assert transformer_block is not None
        assert transformer_block.d_model == 256
        assert transformer_block.n_heads == 8
        assert transformer_block.d_ff == 1024
        assert transformer_block.dropout == 0.1
        
        # Check layer components
        assert hasattr(transformer_block, 'self_attention')
        assert hasattr(transformer_block, 'norm1')
        assert hasattr(transformer_block, 'feed_forward')
        assert hasattr(transformer_block, 'norm2')
    
    def test_happy_path(self, transformer_block, input_tensor):
        """Test normal forward pass succeeds"""
        transformer_block.eval()
        
        output = transformer_block(input_tensor)
        
        # Check output shape
        assert output.shape == input_tensor.shape
        assert output.dtype == input_tensor.dtype
        
        # Check output is not identical to input (transformation occurred)
        assert not torch.allclose(output, input_tensor, atol=1e-6)
        
        # Check output contains no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_edge_cases(self, transformer_block):
        """Test boundary conditions"""
        transformer_block.eval()
        
        # Single sequence
        single_seq = torch.randn(1, 60, 256)
        output = transformer_block(single_seq)
        assert output.shape == single_seq.shape
        
        # Minimal sequence length
        min_seq = torch.randn(1, 1, 256)
        output = transformer_block(min_seq)
        assert output.shape == min_seq.shape
        
        # Large batch
        large_batch = torch.randn(128, 60, 256)
        output = transformer_block(large_batch)
        assert output.shape == large_batch.shape
        
        # Different sequence lengths (if supported)
        diff_seq = torch.randn(8, 120, 256)
        output = transformer_block(diff_seq)
        assert output.shape == diff_seq.shape
    
    def test_error_handling(self, transformer_block):
        """Test error conditions raise appropriately"""
        # Wrong input dimensions
        with pytest.raises((RuntimeError, ValueError)):
            wrong_dims = torch.randn(16, 60)  # Missing feature dimension
            transformer_block(wrong_dims)
        
        # Wrong feature size
        with pytest.raises((RuntimeError, ValueError)):
            wrong_features = torch.randn(16, 60, 128)  # d_model=256 expected
            transformer_block(wrong_features)
        
        # Empty tensor
        with pytest.raises((RuntimeError, ValueError)):
            empty_tensor = torch.empty(0, 0, 256)
            transformer_block(empty_tensor)
    
    def test_gradient_flow(self, transformer_block, input_tensor):
        """Test gradients flow through transformer block"""
        input_tensor.requires_grad_(True)
        transformer_block.train()
        
        output = transformer_block(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()
        assert not torch.isinf(input_tensor.grad).any()
        
        # Check model parameters have gradients
        for param in transformer_block.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_attention_weights_return(self, transformer_block, input_tensor):
        """Test attention weights can be returned when requested"""
        transformer_block.eval()
        
        # Test with return_attention=True if supported
        if hasattr(transformer_block, 'forward') and 'return_attention' in transformer_block.forward.__code__.co_varnames:
            output, attention_weights = transformer_block(input_tensor, return_attention=True)
            
            assert output.shape == input_tensor.shape
            assert attention_weights is not None
            assert attention_weights.shape[0] == input_tensor.shape[0]  # Batch dimension
    
    def test_deterministic_output(self, transformer_block, input_tensor):
        """Test model produces consistent outputs with same input"""
        transformer_block.eval()
        
        torch.manual_seed(42)
        output1 = transformer_block(input_tensor)
        
        torch.manual_seed(42)
        output2 = transformer_block(input_tensor)
        
        assert_close(output1, output2, rtol=1e-5, atol=1e-7)
    
    @pytest.mark.parametrize("n_heads", [1, 4, 8, 16])
    def test_different_head_counts(self, n_heads, input_tensor):
        """Test transformer with different attention head counts"""
        if 256 % n_heads == 0:  # Only test valid head counts
            block = TransformerBlock(d_model=256, n_heads=n_heads)
            block.eval()
            
            output = block(input_tensor)
            assert output.shape == input_tensor.shape
            assert not torch.isnan(output).any()


class TestAttentionPooling:
    """Test suite for AttentionPooling"""
    
    @pytest.fixture
    def attention_pooling(self):
        """Create AttentionPooling instance for testing"""
        return AttentionPooling(d_model=256, num_heads=8)
    
    @pytest.fixture
    def sequence_input(self):
        """Generate sequence input for pooling testing"""
        torch.manual_seed(42)
        return torch.randn(16, 60, 256)  # batch_size, seq_len, d_model
    
    def test_initialization(self, attention_pooling):
        """Test attention pooling initializes correctly"""
        assert attention_pooling is not None
        assert attention_pooling.d_model == 256
        assert attention_pooling.num_heads == 8
    
    def test_happy_path(self, attention_pooling, sequence_input):
        """Test normal pooling operation succeeds"""
        attention_pooling.eval()
        
        output = attention_pooling(sequence_input)
        
        # Should pool sequence dimension
        assert output.shape == (16, 256)  # batch_size, d_model
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_edge_cases(self, attention_pooling):
        """Test boundary conditions"""
        attention_pooling.eval()
        
        # Single timestep
        single_step = torch.randn(1, 1, 256)
        output = attention_pooling(single_step)
        assert output.shape == (1, 256)
        
        # Large sequence
        large_seq = torch.randn(8, 500, 256)
        output = attention_pooling(large_seq)
        assert output.shape == (8, 256)
    
    def test_attention_weights(self, attention_pooling, sequence_input):
        """Test attention weights are computed correctly"""
        attention_pooling.eval()
        
        # Test if attention weights can be returned
        if hasattr(attention_pooling, 'get_attention_weights'):
            weights = attention_pooling.get_attention_weights(sequence_input)
            assert weights.shape[:2] == sequence_input.shape[:2]  # batch_size, seq_len
            
            # Weights should sum to 1 across sequence dimension
            assert torch.allclose(weights.sum(dim=1), torch.ones(sequence_input.shape[0]), atol=1e-6)


class TestPositionalEncoding:
    """Test suite for PositionalEncoding"""
    
    @pytest.fixture
    def positional_encoding(self):
        """Create PositionalEncoding instance for testing"""
        return PositionalEncoding(d_model=256, max_seq_len=100)
    
    def test_initialization(self, positional_encoding):
        """Test positional encoding initializes correctly"""
        assert positional_encoding is not None
        assert positional_encoding.d_model == 256
        assert positional_encoding.max_seq_len == 100
    
    def test_happy_path(self, positional_encoding):
        """Test normal encoding operation succeeds"""
        torch.manual_seed(42)
        input_tensor = torch.randn(16, 60, 256)
        
        output = positional_encoding(input_tensor)
        
        # Shape should be preserved
        assert output.shape == input_tensor.shape
        
        # Output should be different from input (encoding added)
        assert not torch.allclose(output, input_tensor, atol=1e-6)
        
        # No NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_edge_cases(self, positional_encoding):
        """Test boundary conditions"""
        # Single position
        single_pos = torch.randn(1, 1, 256)
        output = positional_encoding(single_pos)
        assert output.shape == single_pos.shape
        
        # Maximum sequence length
        max_seq = torch.randn(1, 100, 256)
        output = positional_encoding(max_seq)
        assert output.shape == max_seq.shape
        
        # Sequence longer than max_seq_len should handle gracefully
        long_seq = torch.randn(1, 150, 256)
        # Should either work or raise appropriate error
        try:
            output = positional_encoding(long_seq)
            assert output.shape == long_seq.shape
        except (RuntimeError, IndexError):
            pass  # Expected behavior for sequences too long
    
    def test_position_consistency(self, positional_encoding):
        """Test same positions get same encodings"""
        # Two identical sequences should get identical positional encodings
        seq1 = torch.zeros(1, 10, 256)
        seq2 = torch.zeros(1, 10, 256)
        
        enc1 = positional_encoding(seq1)
        enc2 = positional_encoding(seq2)
        
        assert_close(enc1, enc2, rtol=1e-7, atol=1e-9)
    
    def test_encoding_properties(self, positional_encoding):
        """Test mathematical properties of positional encoding"""
        # Create zero input to isolate positional encoding
        zero_input = torch.zeros(1, 50, 256)
        encoded = positional_encoding(zero_input)
        
        # Encoding should vary across positions
        pos_encodings = encoded[0]  # Remove batch dimension
        
        # Different positions should have different encodings
        for i in range(min(10, len(pos_encodings) - 1)):
            assert not torch.allclose(pos_encodings[i], pos_encodings[i + 1], atol=1e-6)


class TestPredictionHeads:
    """Test suite for PredictionHeads"""
    
    @pytest.fixture
    def prediction_heads(self):
        """Create PredictionHeads instance for testing"""
        return PredictionHeads(d_model=256, forecast_horizon=5, num_quantiles=5)
    
    @pytest.fixture
    def pooled_input(self):
        """Generate pooled input for prediction heads"""
        torch.manual_seed(42)
        return torch.randn(16, 256)  # batch_size, d_model
    
    def test_initialization(self, prediction_heads):
        """Test prediction heads initialize correctly"""
        assert prediction_heads is not None
        assert prediction_heads.d_model == 256
        assert prediction_heads.forecast_horizon == 5
        assert hasattr(prediction_heads, 'price_head')
        assert hasattr(prediction_heads, 'quantile_heads')
    
    def test_happy_path(self, prediction_heads, pooled_input):
        """Test normal prediction operation succeeds"""
        prediction_heads.eval()
        
        outputs = prediction_heads(pooled_input)
        
        # Check output structure
        assert isinstance(outputs, dict)
        assert 'prices' in outputs
        assert 'quantiles' in outputs
        
        # Check shapes
        assert outputs['prices'].shape == (16, 5)  # batch_size, forecast_horizon
        assert outputs['quantiles'].shape == (16, 5, 5)  # batch_size, forecast_horizon, num_quantiles
        
        # Check no NaN or Inf
        assert not torch.isnan(outputs['prices']).any()
        assert not torch.isnan(outputs['quantiles']).any()
        assert not torch.isinf(outputs['prices']).any()
        assert not torch.isinf(outputs['quantiles']).any()
    
    def test_edge_cases(self, prediction_heads):
        """Test boundary conditions"""
        prediction_heads.eval()
        
        # Single sample
        single_input = torch.randn(1, 256)
        outputs = prediction_heads(single_input)
        assert outputs['prices'].shape == (1, 5)
        assert outputs['quantiles'].shape == (1, 5, 5)
        
        # Large batch
        large_input = torch.randn(128, 256)
        outputs = prediction_heads(large_input)
        assert outputs['prices'].shape == (128, 5)
        assert outputs['quantiles'].shape == (128, 5, 5)
    
    def test_quantile_ordering(self, prediction_heads, pooled_input):
        """Test quantile predictions maintain proper ordering"""
        prediction_heads.eval()
        
        outputs = prediction_heads(pooled_input)
        quantiles = outputs['quantiles']  # shape: (batch, horizon, num_quantiles)
        
        # Check quantile ordering for each sample and time step
        for b in range(quantiles.shape[0]):
            for t in range(quantiles.shape[1]):
                q_values = quantiles[b, t, :]
                # Quantiles should be in non-decreasing order
                for i in range(len(q_values) - 1):
                    assert q_values[i] <= q_values[i + 1] + 1e-6  # Small tolerance for numerical precision
    
    def test_volatility_head(self, prediction_heads, pooled_input):
        """Test volatility head produces positive values"""
        prediction_heads.eval()
        
        outputs = prediction_heads(pooled_input)
        
        if 'volatility' in outputs:
            volatility = outputs['volatility']
            assert volatility.shape == (16, 5)
            # Volatility should be non-negative
            assert (volatility >= 0).all()


class TestInputEmbedding:
    """Test suite for InputEmbedding"""
    
    @pytest.fixture
    def input_embedding(self):
        """Create InputEmbedding instance for testing"""
        return InputEmbedding(input_dim=7, d_model=256)
    
    @pytest.fixture
    def raw_features(self):
        """Generate raw feature input"""
        torch.manual_seed(42)
        return torch.randn(16, 60, 7)  # batch_size, seq_len, features
    
    def test_initialization(self, input_embedding):
        """Test input embedding initializes correctly"""
        assert input_embedding is not None
        assert input_embedding.input_dim == 7
        assert input_embedding.d_model == 256
    
    def test_happy_path(self, input_embedding, raw_features):
        """Test normal embedding operation succeeds"""
        embedded = input_embedding(raw_features)
        
        # Check shape transformation
        assert embedded.shape == (16, 60, 256)
        
        # Check no NaN or Inf
        assert not torch.isnan(embedded).any()
        assert not torch.isinf(embedded).any()
    
    def test_edge_cases(self, input_embedding):
        """Test boundary conditions"""
        # Single feature vector
        single_vec = torch.randn(1, 1, 7)
        embedded = input_embedding(single_vec)
        assert embedded.shape == (1, 1, 256)
        
        # Large sequence
        large_seq = torch.randn(8, 200, 7)
        embedded = input_embedding(large_seq)
        assert embedded.shape == (8, 200, 256)
    
    def test_gradient_flow(self, input_embedding, raw_features):
        """Test gradients flow through embedding"""
        raw_features.requires_grad_(True)
        
        embedded = input_embedding(raw_features)
        loss = embedded.sum()
        loss.backward()
        
        assert raw_features.grad is not None
        assert not torch.isnan(raw_features.grad).any()


class TestTemporalMasking:
    """Test suite for TemporalMasking"""
    
    @pytest.fixture
    def temporal_masking(self):
        """Create TemporalMasking instance for testing"""
        return TemporalMasking()
    
    def test_initialization(self, temporal_masking):
        """Test temporal masking initializes correctly"""
        assert temporal_masking is not None
    
    def test_causal_mask_creation(self, temporal_masking):
        """Test causal mask is created correctly"""
        seq_len = 10
        mask = temporal_masking.create_causal_mask(seq_len)
        
        # Mask should be lower triangular
        assert mask.shape == (seq_len, seq_len)
        
        # Check causal property: can only attend to current and past positions
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[i, j] == 0  # Cannot attend to future
                else:
                    assert mask[i, j] == 1  # Can attend to current and past
    
    def test_padding_mask_creation(self, temporal_masking):
        """Test padding mask handles variable length sequences"""
        batch_size = 4
        seq_len = 10
        lengths = [5, 7, 10, 3]  # Different sequence lengths
        
        mask = temporal_masking.create_padding_mask(batch_size, seq_len, lengths)
        
        assert mask.shape == (batch_size, seq_len)
        
        # Check padding is masked correctly
        for b, length in enumerate(lengths):
            for t in range(seq_len):
                if t < length:
                    assert mask[b, t] == 1  # Valid position
                else:
                    assert mask[b, t] == 0  # Padded position


class TestInterpretableAttention:
    """Test suite for InterpretableAttention"""
    
    @pytest.fixture
    def interpretable_attention(self):
        """Create InterpretableAttention instance for testing"""
        return InterpretableAttention(d_model=256, n_heads=8)
    
    @pytest.fixture
    def attention_input(self):
        """Generate input for attention testing"""
        torch.manual_seed(42)
        return torch.randn(16, 60, 256)
    
    def test_initialization(self, interpretable_attention):
        """Test interpretable attention initializes correctly"""
        assert interpretable_attention is not None
        assert interpretable_attention.d_model == 256
        assert interpretable_attention.n_heads == 8
        assert interpretable_attention.d_k == 256 // 8
    
    def test_happy_path(self, interpretable_attention, attention_input):
        """Test normal attention operation succeeds"""
        interpretable_attention.eval()
        
        output, attention_weights = interpretable_attention(attention_input)
        
        # Check output shape
        assert output.shape == attention_input.shape
        
        # Check attention weights shape
        assert attention_weights.shape == (16, 8, 60, 60)  # batch, heads, seq_len, seq_len
        
        # Attention weights should sum to 1 across key dimension
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones(16, 8, 60),
            atol=1e-5
        )
        
        # Check no NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isnan(attention_weights).any()
    
    def test_attention_properties(self, interpretable_attention, attention_input):
        """Test mathematical properties of attention mechanism"""
        interpretable_attention.eval()
        
        output, attention_weights = interpretable_attention(attention_input)
        
        # Attention weights should be non-negative
        assert (attention_weights >= 0).all()
        
        # Attention weights should sum to 1 for each query position
        weight_sums = attention_weights.sum(dim=-1)
        expected_sums = torch.ones_like(weight_sums)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-5)
    
    def test_masked_attention(self, interpretable_attention, attention_input):
        """Test attention with masking"""
        # Create causal mask
        seq_len = attention_input.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        if 'mask' in interpretable_attention.forward.__code__.co_varnames:
            output, attention_weights = interpretable_attention(attention_input, mask=mask)
            
            # Check that future positions have zero attention
            for h in range(attention_weights.shape[1]):  # For each head
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        assert attention_weights[0, h, i, j] < 1e-6  # Should be ~0


class TestTimeSeriesTransformer:
    """Test suite for complete TimeSeriesTransformer model"""
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing"""
        return {
            'sequence_length': 60,
            'num_features': 7,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'dropout': 0.1,
            'forecast_horizon': 5
        }
    
    @pytest.fixture
    def transformer_model(self, model_config):
        """Create TimeSeriesTransformer instance for testing"""
        return TimeSeriesTransformer(**model_config)
    
    @pytest.fixture
    def model_input(self):
        """Generate model input tensor"""
        torch.manual_seed(42)
        return torch.randn(16, 60, 7)
    
    def test_initialization(self, transformer_model, model_config):
        """Test model initializes correctly"""
        assert transformer_model is not None
        
        # Check configuration
        assert transformer_model.sequence_length == model_config['sequence_length']
        assert transformer_model.num_features == model_config['num_features']
        assert transformer_model.d_model == model_config['d_model']
        
        # Check components exist
        assert hasattr(transformer_model, 'input_embedding')
        assert hasattr(transformer_model, 'positional_encoding')
        assert hasattr(transformer_model, 'transformer_blocks')
        assert hasattr(transformer_model, 'prediction_heads')
    
    def test_happy_path(self, transformer_model, model_input):
        """Test normal model forward pass succeeds"""
        transformer_model.eval()
        
        outputs = transformer_model(model_input)
        
        # Check outputs structure
        assert isinstance(outputs, dict)
        assert 'predictions' in outputs or 'prices' in outputs
        
        # Check output shapes
        if 'predictions' in outputs:
            assert outputs['predictions'].shape == (16, 5)
        if 'prices' in outputs:
            assert outputs['prices'].shape == (16, 5)
            
        # Check no NaN or Inf
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isnan(tensor).any(), f"NaN found in {key}"
                assert not torch.isinf(tensor).any(), f"Inf found in {key}"
    
    def test_edge_cases(self, transformer_model):
        """Test boundary conditions"""
        transformer_model.eval()
        
        # Single sample
        single_input = torch.randn(1, 60, 7)
        outputs = transformer_model(single_input)
        assert list(outputs.values())[0].shape[0] == 1
        
        # Large batch
        large_input = torch.randn(64, 60, 7)
        outputs = transformer_model(large_input)
        assert list(outputs.values())[0].shape[0] == 64
    
    def test_gradient_flow(self, transformer_model, model_input):
        """Test gradients flow through entire model"""
        model_input.requires_grad_(True)
        transformer_model.train()
        
        outputs = transformer_model(model_input)
        
        # Get main prediction output
        main_output = list(outputs.values())[0]
        loss = main_output.sum()
        loss.backward()
        
        # Check input gradients
        assert model_input.grad is not None
        assert not torch.isnan(model_input.grad).any()
        
        # Check model parameter gradients
        for name, param in transformer_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_deterministic(self, transformer_model, model_input):
        """Test model produces consistent outputs"""
        transformer_model.eval()
        
        torch.manual_seed(42)
        output1 = transformer_model(model_input)
        
        torch.manual_seed(42)
        output2 = transformer_model(model_input)
        
        # Compare main outputs
        for key in output1.keys():
            assert_close(output1[key], output2[key], rtol=1e-5, atol=1e-7)
    
    def test_batch_consistency(self, transformer_model):
        """Test batch processing consistency"""
        transformer_model.eval()
        
        # Single input
        single_input = torch.randn(1, 60, 7)
        single_output = transformer_model(single_input)
        
        # Same input in batch
        batch_input = single_input.repeat(5, 1, 1)
        batch_output = transformer_model(batch_input)
        
        # First item in batch should match single output
        for key in single_output.keys():
            assert_close(
                single_output[key],
                batch_output[key][0:1],
                rtol=1e-4,
                atol=1e-6
            )
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_different_batch_sizes(self, transformer_model, batch_size):
        """Test model with different batch sizes"""
        transformer_model.eval()
        
        input_tensor = torch.randn(batch_size, 60, 7)
        outputs = transformer_model(input_tensor)
        
        # All outputs should have correct batch dimension
        for tensor in outputs.values():
            if isinstance(tensor, torch.Tensor):
                assert tensor.shape[0] == batch_size