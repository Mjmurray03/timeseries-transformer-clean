"""
Positional Encoding Implementation

This module implements various positional encoding schemes for transformer models,
including learnable embeddings, sinusoidal encoding, and relative position encoding.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional encoding using trainable parameters.
    
    This approach learns position-specific embeddings during training,
    which can adapt to the specific patterns in time-series data.
    
    Args:
        max_seq_len: Maximum sequence length (default: 60)
        d_model: Model dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        max_seq_len: int = 60,
        d_model: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        
        # Initialize with small random values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings with small random values."""
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        assert d_model == self.d_model, f"Model dimension {d_model} doesn't match expected {self.d_model}"
        
        # Add positional encoding
        pos_encoded = x + self.pos_embedding[:, :seq_len, :]
        
        return self.dropout(pos_encoded)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    Uses sine and cosine functions of different frequencies to encode positions.
    This approach doesn't require training and can handle sequences longer than
    those seen during training.
    
    Args:
        d_model: Model dimension (default: 256)
        max_seq_len: Maximum sequence length (default: 10000)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_seq_len: int = 10000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for frequency calculation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        assert d_model == self.d_model, f"Model dimension {d_model} doesn't match expected {self.d_model}"
        
        # Add positional encoding
        pos_encoded = x + self.pe[:, :seq_len, :]
        
        return self.dropout(pos_encoded)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding that focuses on relative distances between positions.
    
    This approach is particularly useful for time-series data where the relative
    temporal relationships are more important than absolute positions.
    
    Args:
        d_model: Model dimension (default: 256)
        max_relative_position: Maximum relative position to consider (default: 32)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_relative_position: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(dropout)
        
        # Relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize relative position embeddings."""
        nn.init.normal_(self.relative_position_embeddings.weight, mean=0, std=0.02)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position indices.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position indices of shape (seq_len, seq_len)
        """
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip to maximum relative position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to positive indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return final_mat
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add relative positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with relative positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get relative position indices
        relative_positions = self._get_relative_positions(seq_len).to(x.device)
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings(relative_positions)
        
        # Average relative embeddings across sequence dimension
        # This is a simplified approach - more sophisticated methods exist
        relative_encoding = relative_embeddings.mean(dim=0, keepdim=True)
        relative_encoding = relative_encoding.expand(batch_size, -1, -1)
        
        # Add to input
        pos_encoded = x + relative_encoding
        
        return self.dropout(pos_encoded)


class AdaptivePositionalEncoding(nn.Module):
    """
    Adaptive positional encoding that combines multiple encoding schemes.
    
    This approach allows the model to learn which type of positional information
    is most useful for different parts of the sequence.
    
    Args:
        d_model: Model dimension (default: 256)
        max_seq_len: Maximum sequence length (default: 60)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_seq_len: int = 60,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Multiple encoding schemes
        self.learned_encoding = LearnedPositionalEncoding(max_seq_len, d_model, dropout=0.0)
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len * 2, dropout=0.0)
        
        # Gating mechanism to combine encodings
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating network weights."""
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with adaptive positional encoding added
        """
        # Get different encodings (without dropout)
        learned_encoded = self.learned_encoding(x)
        sinusoidal_encoded = self.sinusoidal_encoding(x)
        
        # Compute gating weights based on input
        gate_weights = self.gate(x)  # Shape: (batch_size, seq_len, 2)
        
        # Combine encodings using learned weights
        combined_encoding = (
            gate_weights[:, :, 0:1] * learned_encoded +
            gate_weights[:, :, 1:2] * sinusoidal_encoded
        )
        
        return self.dropout(combined_encoding)


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding specifically designed for time-series data.
    
    This encoding incorporates both absolute time information and relative
    temporal distances, making it suitable for financial time-series.
    
    Args:
        d_model: Model dimension (default: 256)
        max_seq_len: Maximum sequence length (default: 60)
        time_scale: Time scale factor for encoding (default: 1.0)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        max_seq_len: int = 60,
        time_scale: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.time_scale = time_scale
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temporal embeddings
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        
        # Time-aware transformation
        self.time_projection = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize temporal encoding weights."""
        nn.init.normal_(self.temporal_embedding, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.time_projection.weight)
        if self.time_projection.bias is not None:
            nn.init.constant_(self.time_projection.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            time_indices: Optional time indices for each position
            
        Returns:
            Tensor with temporal positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Base temporal encoding
        temporal_encoding = self.temporal_embedding[:, :seq_len, :]
        
        # If time indices provided, modulate encoding
        if time_indices is not None:
            # Scale temporal encoding based on actual time differences
            time_weights = torch.exp(-torch.abs(time_indices) * self.time_scale)
            time_weights = time_weights.unsqueeze(-1)  # Add feature dimension
            temporal_encoding = temporal_encoding * time_weights
        
        # Apply time-aware transformation
        temporal_encoding = self.time_projection(temporal_encoding)
        
        # Add to input
        pos_encoded = x + temporal_encoding
        
        return self.dropout(pos_encoded)


def create_positional_encoding(
    encoding_type: str = "learned",
    d_model: int = 256,
    max_seq_len: int = 60,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of positional encodings.
    
    Args:
        encoding_type: Type of encoding ("learned", "sinusoidal", "relative", "adaptive", "temporal")
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        **kwargs: Additional arguments for specific encoding types
        
    Returns:
        Positional encoding module
    """
    if encoding_type == "learned":
        return LearnedPositionalEncoding(max_seq_len, d_model, dropout)
    elif encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
    elif encoding_type == "relative":
        max_relative_position = kwargs.get("max_relative_position", 32)
        return RelativePositionalEncoding(d_model, max_relative_position, dropout)
    elif encoding_type == "adaptive":
        return AdaptivePositionalEncoding(d_model, max_seq_len, dropout)
    elif encoding_type == "temporal":
        time_scale = kwargs.get("time_scale", 1.0)
        return TemporalPositionalEncoding(d_model, max_seq_len, time_scale, dropout)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")