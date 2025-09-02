"""
Transformer Block Implementation

This module implements the core transformer block with multi-head self-attention,
feed-forward network, residual connections, and layer normalization.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.

    Architecture:
    - Multi-head self-attention with residual connection and layer norm
    - Position-wise feed-forward network with residual connection and layer norm
    - GELU activation function
    - Dropout for regularization

    Args:
        d_model: Model dimension (default: 256)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 1024)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1
    ):
        super().__init__()

        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            If return_attention=True, also returns attention weights
        """
        # Multi-head self-attention with residual connection
        attn_out, attn_weights = self.attention(
            x, x, x, attn_mask=mask, need_weights=return_attention
        )
        x = self.norm1(x + attn_out)

        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_weights
        return x


class InterpretableTransformerBlock(nn.Module):
    """
    Transformer block with enhanced interpretability features.

    This version provides more detailed attention analysis and gradient tracking
    for better model interpretability.
    """

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Separate Q, K, V projections for interpretability
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with interpretable attention computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose for attention computation: (B, H, L, D)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout1(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.W_o(context)

        # First residual connection and layer norm
        x = self.norm1(x + attn_out)

        # Feed-forward network with second residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            # Average attention weights across heads for interpretability
            avg_attention = attention_weights.mean(dim=1)
            return x, avg_attention
        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for autoregressive attention.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for variable length sequences.

    Args:
        lengths: Tensor of sequence lengths
        max_len: Maximum sequence length

    Returns:
        Padding mask of shape (batch_size, max_len)
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask
