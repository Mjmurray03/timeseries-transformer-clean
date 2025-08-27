"""
Temporal masking utilities for preventing future information leakage.

This module implements various masking strategies for time-series transformers,
including causal masking, padding masks, and custom temporal constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class TemporalMaskGenerator:
    """
    Generator for various temporal masking patterns.
    
    Provides static methods to create different types of masks for attention mechanisms
    in time-series transformers, ensuring no future information leakage.
    """
    
    @staticmethod
    def create_causal_mask(
        seq_len: int, 
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create a causal (lower triangular) mask.
        
        Prevents attention to future positions, ensuring autoregressive property.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            dtype: Data type for mask
            
        Returns:
            mask: Causal mask of shape (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
        return mask
    
    @staticmethod
    def create_padding_mask(
        lengths: torch.Tensor,
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Create padding mask for variable-length sequences.
        
        Args:
            lengths: Actual lengths of sequences (batch_size,)
            max_len: Maximum sequence length (optional)
            device: Device to create mask on
            
        Returns:
            mask: Padding mask of shape (batch_size, max_len)
        """
        if max_len is None:
            max_len = lengths.max().item()
        
        batch_size = lengths.shape[0]
        
        # Create range tensor
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        lengths_tensor = lengths.unsqueeze(1)  # (batch_size, 1)
        
        # Create mask
        mask = range_tensor < lengths_tensor  # (batch_size, max_len)
        
        return mask
    
    @staticmethod
    def create_sliding_window_mask(
        seq_len: int,
        window_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create sliding window attention mask.
        
        Each position can only attend to positions within a fixed window.
        
        Args:
            seq_len: Sequence length
            window_size: Size of attention window
            device: Device to create mask on
            dtype: Data type for mask
            
        Returns:
            mask: Sliding window mask of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            end = i + 1  # Include current position
            mask[i, start:end] = 1
        
        return mask
    
    @staticmethod
    def create_block_diagonal_mask(
        seq_len: int,
        block_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create block diagonal attention mask.
        
        Divides sequence into blocks and allows attention within blocks only.
        
        Args:
            seq_len: Sequence length
            block_size: Size of each block
            device: Device to create mask on
            dtype: Data type for mask
            
        Returns:
            mask: Block diagonal mask of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        for i in range(0, seq_len, block_size):
            end = min(i + block_size, seq_len)
            mask[i:end, i:end] = 1
        
        return mask
    
    @staticmethod
    def create_strided_mask(
        seq_len: int,
        stride: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool
    ) -> torch.Tensor:
        """
        Create strided attention mask.
        
        Each position attends to every stride-th previous position.
        
        Args:
            seq_len: Sequence length
            stride: Stride for attention pattern
            device: Device to create mask on
            dtype: Data type for mask
            
        Returns:
            mask: Strided mask of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        for i in range(seq_len):
            # Attend to positions at stride intervals
            for j in range(0, i + 1, stride):
                mask[i, j] = 1
        
        return mask


class AdaptiveTemporalMask(nn.Module):
    """
    Learnable temporal masking with adaptive patterns.
    
    Learns to mask certain temporal positions based on input patterns,
    providing more flexible attention constraints than fixed masks.
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
        mask_ratio: Base masking ratio (default: 0.1)
        temperature: Temperature for Gumbel softmax (default: 1.0)
    """
    
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        mask_ratio: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        
        # Learnable mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Base causal mask (always applied)
        self.register_buffer(
            'causal_mask',
            TemporalMaskGenerator.create_causal_mask(seq_len)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        base_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> torch.Tensor:
        """
        Generate adaptive temporal mask.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            base_mask: Optional base mask to combine with
            training: Whether in training mode
            
        Returns:
            mask: Combined temporal mask (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Predict mask probabilities
        mask_probs = self.mask_predictor(x)  # (batch_size, seq_len, 1)
        mask_probs = mask_probs.squeeze(-1)  # (batch_size, seq_len)
        
        if training:
            # Use Gumbel softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(mask_probs) + 1e-8) + 1e-8)
            mask_logits = torch.log(mask_probs + 1e-8) + gumbel_noise
            adaptive_mask = torch.sigmoid(mask_logits / self.temperature)
        else:
            # Use hard thresholding during inference
            threshold = 1.0 - self.mask_ratio
            adaptive_mask = (mask_probs > threshold).float()
        
        # Expand adaptive mask to attention shape
        adaptive_mask = adaptive_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
        adaptive_mask = adaptive_mask.expand(-1, seq_len, -1)  # (batch_size, seq_len, seq_len)
        
        # Apply causal constraint
        causal_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)  # (1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)
        
        # Combine masks
        combined_mask = adaptive_mask * causal_mask.float()
        
        # Apply base mask if provided
        if base_mask is not None:
            if base_mask.dim() == 2:
                base_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1)
            combined_mask = combined_mask * base_mask.float()
        
        return combined_mask


class TemporalMaskingLayer(nn.Module):
    """
    Complete temporal masking layer with multiple masking strategies.
    
    Combines different masking approaches and provides a unified interface
    for temporal attention constraints in transformer models.
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
        mask_type: Type of masking ("causal", "sliding_window", "adaptive")
        window_size: Window size for sliding window masking
        learnable: Whether to use learnable adaptive masking
    """
    
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        mask_type: str = "causal",
        window_size: Optional[int] = None,
        learnable: bool = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.mask_type = mask_type
        self.window_size = window_size
        self.learnable = learnable
        
        # Create base masks
        if mask_type == "causal":
            self.register_buffer(
                'base_mask',
                TemporalMaskGenerator.create_causal_mask(seq_len)
            )
        elif mask_type == "sliding_window":
            if window_size is None:
                window_size = min(32, seq_len // 2)
            self.register_buffer(
                'base_mask',
                TemporalMaskGenerator.create_sliding_window_mask(seq_len, window_size)
            )
        elif mask_type == "block_diagonal":
            block_size = window_size or 16
            self.register_buffer(
                'base_mask',
                TemporalMaskGenerator.create_block_diagonal_mask(seq_len, block_size)
            )
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Add adaptive masking if requested
        if learnable:
            self.adaptive_mask = AdaptiveTemporalMask(seq_len, d_model)
        else:
            self.adaptive_mask = None
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        custom_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate complete temporal mask.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            padding_mask: Optional padding mask (batch_size, seq_len)
            custom_mask: Optional custom mask to combine
            
        Returns:
            mask: Complete temporal mask (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Start with base mask
        mask = self.base_mask[:seq_len, :seq_len].unsqueeze(0)  # (1, seq_len, seq_len)
        mask = mask.expand(batch_size, -1, -1).float()  # (batch_size, seq_len, seq_len)
        
        # Apply adaptive masking if available
        if self.adaptive_mask is not None:
            adaptive_mask = self.adaptive_mask(x, mask, training=self.training)
            mask = mask * adaptive_mask
        
        # Apply padding mask
        if padding_mask is not None:
            # Expand padding mask to attention shape
            padding_mask = padding_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            padding_mask = padding_mask.expand(-1, seq_len, -1)  # (batch_size, seq_len, seq_len)
            mask = mask * padding_mask.float()
        
        # Apply custom mask
        if custom_mask is not None:
            if custom_mask.dim() == 2:
                custom_mask = custom_mask.unsqueeze(0).expand(batch_size, -1, -1)
            mask = mask * custom_mask.float()
        
        return mask
    
    def get_mask_info(self) -> dict:
        """Get information about the masking configuration."""
        info = {
            'mask_type': self.mask_type,
            'seq_len': self.seq_len,
            'learnable': self.learnable,
            'window_size': self.window_size
        }
        
        if hasattr(self, 'base_mask'):
            # Compute mask statistics
            base_mask = self.base_mask.float()
            info['sparsity'] = 1.0 - (base_mask.sum() / base_mask.numel()).item()
            info['avg_attention_span'] = base_mask.sum(dim=-1).float().mean().item()
        
        return info


class FutureLeakageDetector:
    """
    Utility to detect potential future information leakage in attention patterns.
    
    Analyzes attention weights to identify cases where the model might be
    inadvertently accessing future information.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize leakage detector.
        
        Args:
            tolerance: Tolerance for detecting non-zero future attention
        """
        self.tolerance = tolerance
    
    def detect_leakage(
        self,
        attention_weights: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Detect future information leakage in attention weights.
        
        Args:
            attention_weights: Attention weights (B, H, L, L) or (B, L, L)
            causal_mask: Optional causal mask for reference
            
        Returns:
            leakage_info: Dictionary containing leakage analysis
        """
        if attention_weights.dim() == 3:
            # Add head dimension
            attention_weights = attention_weights.unsqueeze(1)
        
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Create causal mask if not provided
        if causal_mask is None:
            causal_mask = TemporalMaskGenerator.create_causal_mask(
                seq_len, device=attention_weights.device
            )
        
        # Identify future positions (upper triangular part)
        future_mask = ~causal_mask  # Invert causal mask
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        
        # Extract future attention weights
        future_attention = attention_weights * future_mask.float()
        
        # Compute leakage metrics
        total_future_attention = future_attention.sum(dim=(-2, -1))  # (B, H)
        max_future_attention = future_attention.max(dim=-1)[0].max(dim=-1)[0]  # (B, H)
        
        # Count positions with significant future attention
        significant_leakage = (future_attention > self.tolerance).sum(dim=(-2, -1))  # (B, H)
        
        # Aggregate across batch and heads
        leakage_info = {
            'total_leakage': total_future_attention.mean().item(),
            'max_leakage': max_future_attention.mean().item(),
            'leakage_positions': significant_leakage.float().mean().item(),
            'leakage_ratio': (total_future_attention.sum(dim=-1) / 
                            attention_weights.sum(dim=(-2, -1))).mean().item(),
            'has_leakage': (max_future_attention > self.tolerance).any().item()
        }
        
        return leakage_info
    
    def validate_causal_attention(
        self,
        attention_weights: torch.Tensor,
        strict: bool = True
    ) -> bool:
        """
        Validate that attention weights respect causal constraints.
        
        Args:
            attention_weights: Attention weights to validate
            strict: Whether to use strict validation (no tolerance)
            
        Returns:
            is_valid: Whether attention is properly causal
        """
        leakage_info = self.detect_leakage(attention_weights)
        
        if strict:
            return not leakage_info['has_leakage']
        else:
            return leakage_info['max_leakage'] < self.tolerance