"""
Unit tests for temporal masking components.
"""

import pytest
import torch
import torch.nn as nn
from src.models.components.temporal_masking import (
    TemporalMaskGenerator,
    AdaptiveTemporalMask,
    TemporalMaskingLayer,
    FutureLeakageDetector
)


class TestTemporalMaskGenerator:
    """Test suite for TemporalMaskGenerator"""
    
    def test_create_causal_mask(self):
        """Test causal mask creation"""
        seq_len = 10
        mask = TemporalMaskGenerator.create_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check lower triangular property
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == True
                else:
                    assert mask[i, j] == False
    
    def test_create_padding_mask(self):
        """Test padding mask creation"""
        lengths = torch.tensor([5, 8, 3, 10])
        max_len = 10
        
        mask = TemporalMaskGenerator.create_padding_mask(lengths, max_len)
        
        assert mask.shape == (4, max_len)
        assert mask.dtype == torch.bool
        
        # Check padding correctness
        assert mask[0, :5].all()  # First 5 positions should be True
        assert not mask[0, 5:].any()  # Rest should be False
        assert mask[1, :8].all()
        assert not mask[1, 8:].any()
        assert mask[2, :3].all()
        assert not mask[2, 3:].any()
        assert mask[3, :].all()  # All positions should be True
    
    def test_create_sliding_window_mask(self):
        """Test sliding window mask creation"""
        seq_len = 10
        window_size = 3
        
        mask = TemporalMaskGenerator.create_sliding_window_mask(seq_len, window_size)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check window property
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            end = i + 1
            
            # Positions within window should be True
            assert mask[i, start:end].all()
            
            # Positions outside window should be False
            if start > 0:
                assert not mask[i, :start].any()
            if end < seq_len:
                assert not mask[i, end:].any()
    
    def test_create_block_diagonal_mask(self):
        """Test block diagonal mask creation"""
        seq_len = 12
        block_size = 4
        
        mask = TemporalMaskGenerator.create_block_diagonal_mask(seq_len, block_size)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check block structure
        n_blocks = seq_len // block_size
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = start + block_size
            
            # Within block should be True
            assert mask[start:end, start:end].all()
            
            # Outside block should be False (check a few positions)
            if start > 0:
                assert not mask[start, start-1]
            if end < seq_len:
                assert not mask[start, end]
    
    def test_create_strided_mask(self):
        """Test strided mask creation"""
        seq_len = 10
        stride = 2
        
        mask = TemporalMaskGenerator.create_strided_mask(seq_len, stride)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check stride pattern
        for i in range(seq_len):
            for j in range(0, i + 1, stride):
                assert mask[i, j] == True
            
            # Check non-stride positions are False
            for j in range(i + 1):
                if j % stride != 0:
                    assert mask[i, j] == False


class TestAdaptiveTemporalMask:
    """Test suite for AdaptiveTemporalMask"""
    
    @pytest.fixture
    def adaptive_mask(self):
        """Create adaptive temporal mask for testing"""
        return AdaptiveTemporalMask(seq_len=60, d_model=256, mask_ratio=0.1)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)
    
    def test_initialization(self, adaptive_mask):
        """Test adaptive mask initializes correctly"""
        assert adaptive_mask.seq_len == 60
        assert adaptive_mask.d_model == 256
        assert adaptive_mask.mask_ratio == 0.1
        assert adaptive_mask.temperature == 1.0
        assert isinstance(adaptive_mask.mask_predictor, nn.Sequential)
        assert adaptive_mask.causal_mask.shape == (60, 60)
    
    def test_forward_training(self, adaptive_mask, sample_input):
        """Test forward pass in training mode"""
        adaptive_mask.train()
        mask = adaptive_mask(sample_input, training=True)
        
        assert mask.shape == (4, 60, 60)
        assert mask.dtype == torch.float32
        assert (mask >= 0).all()
        assert (mask <= 1).all()
        
        # Check causal property (upper triangular should be zero)
        for i in range(60):
            for j in range(i + 1, 60):
                assert torch.allclose(mask[:, i, j], torch.zeros(4), atol=1e-6)
    
    def test_forward_inference(self, adaptive_mask, sample_input):
        """Test forward pass in inference mode"""
        adaptive_mask.eval()
        mask = adaptive_mask(sample_input, training=False)
        
        assert mask.shape == (4, 60, 60)
        assert mask.dtype == torch.float32
        
        # In inference mode, mask should be binary
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2  # Should be mostly 0s and 1s
        assert 0.0 in unique_values
    
    def test_gradient_flow(self, adaptive_mask, sample_input):
        """Test gradients flow through adaptive mask"""
        sample_input.requires_grad_(True)
        mask = adaptive_mask(sample_input)
        loss = mask.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()


class TestTemporalMaskingLayer:
    """Test suite for TemporalMaskingLayer"""
    
    @pytest.fixture
    def masking_layer(self):
        """Create temporal masking layer for testing"""
        return TemporalMaskingLayer(
            seq_len=60, 
            d_model=256, 
            mask_type="causal",
            learnable=False
        )
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensor"""
        return torch.randn(4, 60, 256)
    
    def test_initialization_causal(self):
        """Test initialization with causal masking"""
        layer = TemporalMaskingLayer(seq_len=60, d_model=256, mask_type="causal")
        
        assert layer.seq_len == 60
        assert layer.d_model == 256
        assert layer.mask_type == "causal"
        assert layer.adaptive_mask is None
        assert layer.base_mask.shape == (60, 60)
    
    def test_initialization_sliding_window(self):
        """Test initialization with sliding window masking"""
        layer = TemporalMaskingLayer(
            seq_len=60, 
            d_model=256, 
            mask_type="sliding_window",
            window_size=10
        )
        
        assert layer.mask_type == "sliding_window"
        assert layer.window_size == 10
        assert layer.base_mask.shape == (60, 60)
    
    def test_initialization_learnable(self):
        """Test initialization with learnable masking"""
        layer = TemporalMaskingLayer(
            seq_len=60, 
            d_model=256, 
            mask_type="causal",
            learnable=True
        )
        
        assert layer.learnable == True
        assert layer.adaptive_mask is not None
        assert isinstance(layer.adaptive_mask, AdaptiveTemporalMask)
    
    def test_forward_basic(self, masking_layer, sample_input):
        """Test basic forward pass"""
        mask = masking_layer(sample_input)
        
        assert mask.shape == (4, 60, 60)
        assert mask.dtype == torch.float32
        assert (mask >= 0).all()
        assert (mask <= 1).all()
    
    def test_forward_with_padding_mask(self, masking_layer, sample_input):
        """Test forward pass with padding mask"""
        padding_mask = torch.ones(4, 60)
        padding_mask[0, 50:] = 0  # Mask last 10 positions for first sample
        padding_mask[1, 55:] = 0  # Mask last 5 positions for second sample
        
        mask = masking_layer(sample_input, padding_mask=padding_mask)
        
        assert mask.shape == (4, 60, 60)
        
        # Check padding is applied
        assert torch.allclose(mask[0, :, 50:], torch.zeros(60, 10), atol=1e-6)
        assert torch.allclose(mask[1, :, 55:], torch.zeros(60, 5), atol=1e-6)
    
    def test_get_mask_info(self, masking_layer):
        """Test mask information retrieval"""
        info = masking_layer.get_mask_info()
        
        required_keys = ['mask_type', 'seq_len', 'learnable', 'window_size', 'sparsity', 'avg_attention_span']
        for key in required_keys:
            assert key in info
        
        assert info['mask_type'] == 'causal'
        assert info['seq_len'] == 60
        assert info['learnable'] == False
        assert 0 <= info['sparsity'] <= 1
        assert info['avg_attention_span'] > 0
    
    def test_invalid_mask_type(self):
        """Test error handling for invalid mask type"""
        with pytest.raises(ValueError):
            TemporalMaskingLayer(seq_len=60, d_model=256, mask_type="invalid")


class TestFutureLeakageDetector:
    """Test suite for FutureLeakageDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create future leakage detector for testing"""
        return FutureLeakageDetector(tolerance=1e-6)
    
    @pytest.fixture
    def causal_attention(self):
        """Generate properly causal attention weights"""
        seq_len = 10
        attention = torch.randn(2, 4, seq_len, seq_len)
        
        # Apply causal mask
        causal_mask = TemporalMaskGenerator.create_causal_mask(seq_len)
        attention = attention.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attention = torch.softmax(attention, dim=-1)
        
        return attention
    
    @pytest.fixture
    def leaky_attention(self):
        """Generate attention weights with future leakage"""
        seq_len = 10
        attention = torch.randn(2, 4, seq_len, seq_len)
        attention = torch.softmax(attention, dim=-1)  # No causal masking
        
        return attention
    
    def test_detect_no_leakage(self, detector, causal_attention):
        """Test detection with properly causal attention"""
        leakage_info = detector.detect_leakage(causal_attention)
        
        required_keys = [
            'total_leakage', 'max_leakage', 'leakage_positions',
            'leakage_ratio', 'has_leakage'
        ]
        
        for key in required_keys:
            assert key in leakage_info
        
        # Should detect no leakage
        assert leakage_info['total_leakage'] < detector.tolerance
        assert leakage_info['max_leakage'] < detector.tolerance
        assert leakage_info['has_leakage'] == False
    
    def test_detect_leakage(self, detector, leaky_attention):
        """Test detection with future information leakage"""
        leakage_info = detector.detect_leakage(leaky_attention)
        
        # Should detect leakage
        assert leakage_info['total_leakage'] > detector.tolerance
        assert leakage_info['max_leakage'] > detector.tolerance
        assert leakage_info['has_leakage'] == True
        assert leakage_info['leakage_positions'] > 0
    
    def test_validate_causal_attention(self, detector, causal_attention, leaky_attention):
        """Test causal attention validation"""
        # Causal attention should be valid
        assert detector.validate_causal_attention(causal_attention, strict=True)
        assert detector.validate_causal_attention(causal_attention, strict=False)
        
        # Leaky attention should be invalid
        assert not detector.validate_causal_attention(leaky_attention, strict=True)
        assert not detector.validate_causal_attention(leaky_attention, strict=False)
    
    def test_3d_attention_handling(self, detector):
        """Test handling of 3D attention weights"""
        # Create 3D attention (no head dimension)
        attention_3d = torch.randn(2, 10, 10)
        attention_3d = torch.softmax(attention_3d, dim=-1)
        
        leakage_info = detector.detect_leakage(attention_3d)
        
        # Should handle 3D input correctly
        assert isinstance(leakage_info, dict)
        assert 'has_leakage' in leakage_info