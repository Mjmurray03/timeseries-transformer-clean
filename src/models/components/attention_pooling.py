"""
Attention pooling implementation for sequence aggregation.

This module implements attention-based pooling mechanisms that aggregate
sequence representations into fixed-size vectors for downstream prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer that aggregates sequence representations.
    
    Uses a learnable query vector to compute attention weights over the sequence,
    then performs weighted average pooling to produce a fixed-size representation.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability (default: 0.1)
        temperature: Temperature scaling for attention weights (default: 1.0)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # Learnable query vector for attention computation
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Linear transformation for keys
        self.key_projection = nn.Linear(d_model, d_model)
        
        # Optional value transformation
        self.value_projection = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            pooled: Pooled representation of shape (batch_size, d_model)
            attention_weights: Optional attention weights (batch_size, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply layer normalization
        x_norm = self.layer_norm(x)
        
        # Compute keys and values
        keys = self.key_projection(x_norm)  # (B, L, D)
        values = self.value_projection(x_norm)  # (B, L, D)
        
        # Expand query for batch computation
        query = self.query.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        query = query.expand(batch_size, 1, d_model)  # (B, 1, D)
        
        # Compute attention scores
        scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, L)
        scores = scores.squeeze(1) / (math.sqrt(d_model) * self.temperature)  # (B, L)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, L)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_weights_expanded = attention_weights.unsqueeze(1)  # (B, 1, L)
        pooled = torch.bmm(attention_weights_expanded, values)  # (B, 1, D)
        pooled = pooled.squeeze(1)  # (B, D)
        
        if return_attention:
            return pooled, attention_weights
        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for richer representation aggregation.
    
    Uses multiple attention heads with different learnable queries to capture
    different aspects of the sequence for pooling.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        temperature: Temperature scaling for attention weights (default: 1.0)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 8, 
        dropout: float = 0.1, 
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Learnable query vectors for each head
        self.queries = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.02)
        
        # Linear projections
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            pooled: Pooled representation of shape (batch_size, d_model)
            attention_weights: Optional attention weights (batch_size, n_heads, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply layer normalization
        x_norm = self.layer_norm(x)
        
        # Compute keys and values
        keys = self.key_projection(x_norm)  # (B, L, D)
        values = self.value_projection(x_norm)  # (B, L, D)
        
        # Reshape for multi-head computation
        keys = keys.view(batch_size, seq_len, self.n_heads, self.d_k)  # (B, L, H, D_k)
        values = values.view(batch_size, seq_len, self.n_heads, self.d_k)  # (B, L, H, D_k)
        
        # Transpose for efficient computation
        keys = keys.transpose(1, 2)  # (B, H, L, D_k)
        values = values.transpose(1, 2)  # (B, H, L, D_k)
        
        # Expand queries for batch computation
        queries = self.queries.unsqueeze(0).unsqueeze(2)  # (1, H, 1, D_k)
        queries = queries.expand(batch_size, self.n_heads, 1, self.d_k)  # (B, H, 1, D_k)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, H, 1, L)
        scores = scores.squeeze(2) / (math.sqrt(self.d_k) * self.temperature)  # (B, H, L)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, self.n_heads, -1)  # (B, H, L)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, L)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_weights_expanded = attention_weights.unsqueeze(2)  # (B, H, 1, L)
        pooled = torch.matmul(attention_weights_expanded, values)  # (B, H, 1, D_k)
        pooled = pooled.squeeze(2)  # (B, H, D_k)
        
        # Concatenate heads
        pooled = pooled.transpose(1, 2).contiguous()  # (B, D_k, H)
        pooled = pooled.view(batch_size, d_model)  # (B, D)
        
        # Final output projection
        pooled = self.output_projection(pooled)
        
        if return_attention:
            return pooled, attention_weights
        return pooled


class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical attention pooling that first pools local windows, then globally.
    
    This approach can capture both local and global patterns in the sequence
    while being more computationally efficient for long sequences.
    
    Args:
        d_model: Model dimension
        window_size: Size of local windows for first-level pooling
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Local attention pooling
        self.local_pooling = AttentionPooling(d_model, dropout)
        
        # Global attention pooling
        self.global_pooling = AttentionPooling(d_model, dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of hierarchical attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            pooled: Pooled representation of shape (batch_size, d_model)
            attention_weights: Optional tuple of (local_weights, global_weights)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Pad sequence to be divisible by window_size
        pad_len = (self.window_size - seq_len % self.window_size) % self.window_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask_padded = F.pad(mask, (0, pad_len), value=0)
            else:
                mask_padded = None
        else:
            x_padded = x
            mask_padded = mask
        
        # Reshape into windows
        padded_len = x_padded.shape[1]
        n_windows = padded_len // self.window_size
        
        x_windows = x_padded.view(
            batch_size, n_windows, self.window_size, d_model
        )  # (B, W, S, D)
        
        if mask_padded is not None:
            mask_windows = mask_padded.view(
                batch_size, n_windows, self.window_size
            )  # (B, W, S)
        else:
            mask_windows = None
        
        # Local pooling within each window
        local_pooled = []
        local_attention_weights = []
        
        for i in range(n_windows):
            window = x_windows[:, i, :, :]  # (B, S, D)
            window_mask = mask_windows[:, i, :] if mask_windows is not None else None
            
            if return_attention:
                pooled, attn_weights = self.local_pooling(
                    window, window_mask, return_attention=True
                )
                local_attention_weights.append(attn_weights)
            else:
                pooled = self.local_pooling(window, window_mask)
            
            local_pooled.append(pooled)
        
        # Stack local pooled representations
        local_pooled = torch.stack(local_pooled, dim=1)  # (B, W, D)
        
        # Global pooling across windows
        if return_attention:
            global_pooled, global_attention_weights = self.global_pooling(
                local_pooled, return_attention=True
            )
            
            # Combine attention weights
            local_attention_weights = torch.stack(local_attention_weights, dim=1)  # (B, W, S)
            attention_weights = (local_attention_weights, global_attention_weights)
            
            return global_pooled, attention_weights
        else:
            global_pooled = self.global_pooling(local_pooled)
            return global_pooled