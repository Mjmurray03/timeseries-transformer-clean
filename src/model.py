"""
Time-Series Transformer Model Implementation

A clean, academic-ready implementation of transformer-based stock price forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np


class InputEmbedding(nn.Module):
    """Linear input embedding with layer normalization."""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to model dimension."""
        embedded = self.projection(x)  # (batch, seq_len, d_model)
        embedded = self.layer_norm(embedded)
        return self.dropout(embedded)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence modeling."""
    
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(max_seq_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        return self.dropout(x + pos_encoding)


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with multi-head attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections and layer normalization."""
        # Multi-head attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based sequence pooling for final representation."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence using attention weights."""
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)
        
        return self.dropout(pooled)


class TimeSeriesTransformer(nn.Module):
    """
    Time-Series Transformer for Stock Price Forecasting
    
    Architecture:
    - Input embedding with layer normalization
    - Learnable positional encoding
    - Multi-layer transformer encoder
    - Attention pooling
    - Linear output projection
    
    Args:
        input_dim: Number of input features (default: 8)
        hidden_dim: Model dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout probability (default: 0.1)
        max_seq_length: Maximum sequence length (default: 60)
        output_dim: Output dimension (default: 3 for 3-day forecast)
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 60,
        output_dim: int = 3
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.output_dim = output_dim
        
        # Validation
        assert hidden_dim % num_heads == 0, f"hidden_dim must be divisible by num_heads"
        
        # Components
        self.input_embedding = InputEmbedding(input_dim, hidden_dim, dropout)
        self.positional_encoding = PositionalEncoding(max_seq_length, hidden_dim, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.attention_pooling = AttentionPooling(hidden_dim, dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Optional padding mask (batch_size, seq_len)
            
        Returns:
            Predictions (batch_size, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Validate input dimensions
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        assert seq_len <= self.max_seq_length, f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_length}"
        
        # 1. Input embedding
        embedded = self.input_embedding(x)  # (batch, seq_len, hidden_dim)
        
        # 2. Positional encoding
        encoded = self.positional_encoding(embedded)  # (batch, seq_len, hidden_dim)
        
        # 3. Transformer layers
        hidden_states = encoded
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, mask=mask)
        
        # 4. Attention pooling
        pooled = self.attention_pooling(hidden_states, mask=mask)  # (batch, hidden_dim)
        
        # 5. Output projection
        output = self.output_layer(pooled)  # (batch, output_dim)
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model configuration and statistics."""
        return {
            'architecture': 'TimeSeriesTransformer',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_length': self.max_seq_length,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'total_parameters': self.count_parameters()
        }