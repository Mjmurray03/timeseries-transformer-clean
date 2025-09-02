"""
Unit tests for interpretable attention components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch
from src.models.components.interpretable_attention import (
    InterpretableAttention,
    AttentionVisualizer,
    AttentionAnalyzer
)


class TestInterpretableAttention:
    """Test suite for InterpretableAttention"""
    
    @pytest.fixture
    def attention_layer(self):
        """Create interpretable attention layer for testing"""
        return InterpretableAttention(d_model=256, n_heads=8, dropout=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)  # (batch_size, seq_len, d_model)
    
    @pytest.fixture
    def sample_mask(self):
        """Generate sample causal mask"""
        seq_len = 60
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
        return mask.unsqueeze(0).expand(4, -1, -1)  # (batch_size, seq_len, seq_len)
    
    def test_initialization(self, attention_layer):
        """Test interpretable attention initializes correctly"""
        assert attention_layer.d_model == 256
        assert attention_layer.n_heads == 8
        assert attention_layer.d_k == 32  # 256 // 8
        assert attention_layer.temperature == 1.0
        assert isinstance(attention_layer.W_q, nn.Linear)
        assert isinstance(attention_layer.W_k, nn.Linear)
        assert isinstance(attention_layer.W_v, nn.Linear)
        assert isinstance(attention_layer.W_o, nn.Linear)
    
    def test_forward_pass(self, attention_layer, sample_input):
        """Test forward pass produces correct output shape"""
        output = attention_layer(sample_input)
        
        assert output.shape == (4, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_mask(self, attention_layer, sample_input, sample_mask):
        """Test forward pass with causal mask"""
        output = attention_layer(sample_input, mask=sample_mask)
        
        assert output.shape == (4, 60, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_return_attention_weights(self, attention_layer, sample_input):
        """Test returning attention weights"""
        output, attention_weights = attention_layer(
            sample_input, return_attention=True
        )
        
        assert output.shape == (4, 60, 256)
        assert attention_weights.shape == (4, 8, 60, 60)  # (B, H, L, L)
        
        # Check attention weights sum to 1 along last dimension
        assert torch.allclose(
            attention_weights.sum(dim=-1), 
            torch.ones(4, 8, 60), 
            atol=1e-6
        )
        
        # Check attention weights are non-negative
        assert (attention_weights >= 0).all()
    
    def test_attention_storage(self, attention_layer, sample_input):
        """Test attention weight storage functionality"""
        # Initially no stored attention
        assert attention_layer.get_attention_weights() is None
        assert attention_layer.get_head_importance() is None
        
        # Forward pass with storage
        output = attention_layer(sample_input, store_attention=True)
        
        # Check stored attention
        stored_attention = attention_layer.get_attention_weights()
        head_importance = attention_layer.get_head_importance()
        
        assert stored_attention is not None
        assert stored_attention.shape == (4, 8, 60, 60)
        assert head_importance is not None
        assert head_importance.shape == (8,)
        assert torch.allclose(head_importance.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_head_importance_computation(self, attention_layer, sample_input):
        """Test head importance computation"""
        attention_layer(sample_input, store_attention=True)
        head_importance = attention_layer.get_head_importance()
        
        # Check properties
        assert (head_importance > 0).all()  # All positive
        assert torch.allclose(head_importance.sum(), torch.tensor(1.0), atol=1e-6)  # Normalized
    
    def test_attention_aggregation(self, attention_layer, sample_input):
        """Test attention head aggregation methods"""
        attention_layer(sample_input, store_attention=True)
        
        # Test different aggregation methods
        methods = ["mean", "max", "weighted", "top_k"]
        
        for method in methods:
            aggregated = attention_layer.aggregate_attention_heads(method=method)
            assert aggregated is not None
            assert aggregated.shape == (4, 60, 60)  # (B, L, L)
            assert (aggregated >= 0).all()
    
    def test_gradient_flow(self, attention_layer, sample_input):
        """Test gradients flow through the layer"""
        sample_input.requires_grad_(True)
        output = attention_layer(sample_input)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert not torch.isinf(sample_input.grad).any()


class TestAttentionVisualizer:
    """Test suite for AttentionVisualizer"""
    
    @pytest.fixture
    def visualizer(self):
        """Create attention visualizer for testing"""
        return AttentionVisualizer()
    
    @pytest.fixture
    def sample_attention(self):
        """Generate sample attention weights"""
        return torch.randn(60, 60).softmax(dim=-1)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_attention_heatmap(self, mock_show, mock_savefig, visualizer, sample_attention):
        """Test attention heatmap plotting"""
        fig = visualizer.plot_attention_heatmap(sample_attention)
        
        assert fig is not None
        # Test with save path
        visualizer.plot_attention_heatmap(sample_attention, save_path="test.png")
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_temporal_attention(self, mock_show, mock_savefig, visualizer, sample_attention):
        """Test temporal attention plotting"""
        fig = visualizer.plot_temporal_attention(sample_attention, query_position=30)
        
        assert fig is not None
        # Test with save path
        visualizer.plot_temporal_attention(
            sample_attention, query_position=30, save_path="test.png"
        )
        mock_savefig.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_head_importance(self, mock_show, mock_savefig, visualizer):
        """Test head importance plotting"""
        head_importance = torch.rand(8)
        head_importance = head_importance / head_importance.sum()
        
        fig = visualizer.plot_head_importance(head_importance)
        
        assert fig is not None
        # Test with save path
        visualizer.plot_head_importance(head_importance, save_path="test.png")
        mock_savefig.assert_called()


class TestAttentionAnalyzer:
    """Test suite for AttentionAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create attention analyzer for testing"""
        return AttentionAnalyzer()
    
    @pytest.fixture
    def sample_attention_4d(self):
        """Generate sample 4D attention weights"""
        attention = torch.randn(4, 8, 60, 60)
        return attention.softmax(dim=-1)
    
    def test_compute_attention_entropy(self, analyzer, sample_attention_4d):
        """Test attention entropy computation"""
        entropy = analyzer.compute_attention_entropy(sample_attention_4d)
        
        assert entropy.shape == (4, 8, 60)  # (B, H, L)
        assert (entropy >= 0).all()  # Entropy is non-negative
        assert not torch.isnan(entropy).any()
    
    def test_detect_attention_anomalies(self, analyzer, sample_attention_4d):
        """Test attention anomaly detection"""
        anomalies = analyzer.detect_attention_anomalies(sample_attention_4d)
        
        required_keys = [
            'entropy_anomalies', 'attention_anomalies',
            'entropy_zscore', 'max_attention_zscore'
        ]
        
        for key in required_keys:
            assert key in anomalies
        
        # Check shapes
        assert anomalies['entropy_anomalies'].shape == (4, 8, 60)
        assert anomalies['attention_anomalies'].shape == (4, 8, 60)
        
        # Check boolean types for anomaly flags
        assert anomalies['entropy_anomalies'].dtype == torch.bool
        assert anomalies['attention_anomalies'].dtype == torch.bool
    
    def test_analyze_temporal_patterns(self, analyzer, sample_attention_4d):
        """Test temporal pattern analysis"""
        patterns = analyzer.analyze_temporal_patterns(sample_attention_4d)
        
        required_keys = [
            'average_locality', 'recency_bias',
            'diagonal_dominance', 'attention_spread'
        ]
        
        for key in required_keys:
            assert key in patterns
            assert isinstance(patterns[key], float)
            assert not np.isnan(patterns[key])
        
        # Check reasonable ranges
        assert 0 <= patterns['recency_bias'] <= 1
        assert 0 <= patterns['diagonal_dominance'] <= 1
        assert patterns['attention_spread'] >= 0