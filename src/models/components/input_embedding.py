"""
Input Embedding Layer Implementation

This module implements input embedding layers for transforming raw time-series features
into high-dimensional representations suitable for transformer processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


class InputEmbedding(nn.Module):
    """
    Basic input embedding layer for time-series data.
    
    Transforms input features from input_dim to d_model dimensions with
    layer normalization and dropout regularization.
    
    Args:
        input_dim: Input feature dimension (default: 7 for OHLCV + Returns + Volume_Ratio)
        d_model: Model dimension (default: 256)
        dropout: Dropout probability (default: 0.1)
        use_layer_norm: Whether to apply layer normalization (default: True)
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Linear projection to model dimension
        self.projection = nn.Linear(input_dim, d_model)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through input embedding layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        
        # Linear projection
        embedded = self.projection(x)
        
        # Apply layer normalization
        if self.use_layer_norm:
            embedded = self.layer_norm(embedded)
        
        # Apply dropout
        embedded = self.dropout_layer(embedded)
        
        return embedded


class FeatureWiseEmbedding(nn.Module):
    """
    Feature-wise embedding that processes each input feature separately.
    
    This approach allows the model to learn different transformations for
    different types of features (price, volume, technical indicators).
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        feature_names: Optional list of feature names for interpretability
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        feature_names: Optional[List[str]] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(input_dim)]
        
        assert len(self.feature_names) == input_dim, "Feature names must match input dimension"
        
        # Calculate embedding dimension per feature
        self.feature_embed_dim = d_model // input_dim
        
        # Separate embedding for each feature
        self.feature_embeddings = nn.ModuleDict({
            name: nn.Linear(1, self.feature_embed_dim)
            for name in self.feature_names
        })
        
        # Final projection to ensure correct output dimension
        total_embed_dim = self.feature_embed_dim * input_dim
        self.final_projection = nn.Linear(total_embed_dim, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all feature embeddings."""
        for embedding in self.feature_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
            if embedding.bias is not None:
                nn.init.constant_(embedding.bias, 0)
        
        nn.init.xavier_uniform_(self.final_projection.weight)
        if self.final_projection.bias is not None:
            nn.init.constant_(self.final_projection.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature-wise embedding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        
        # Process each feature separately
        feature_embeddings = []
        for i, name in enumerate(self.feature_names):
            feature_data = x[:, :, i:i+1]  # Shape: (batch_size, seq_len, 1)
            feature_embedded = self.feature_embeddings[name](feature_data)
            feature_embeddings.append(feature_embedded)
        
        # Concatenate feature embeddings
        concatenated = torch.cat(feature_embeddings, dim=-1)
        
        # Final projection to ensure correct dimension
        embedded = self.final_projection(concatenated)
        
        # Apply layer normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class ScaledInputEmbedding(nn.Module):
    """
    Input embedding with learnable scaling factors.
    
    This approach learns optimal scaling for the embedding to improve
    training stability and convergence.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        scale_factor: Initial scaling factor (default: sqrt(d_model))
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        scale_factor: Optional[float] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Default scale factor is sqrt(d_model) as in original Transformer
        if scale_factor is None:
            scale_factor = math.sqrt(d_model)
        
        # Learnable scaling parameter
        self.scale = nn.Parameter(torch.tensor(scale_factor))
        
        # Linear projection
        self.projection = nn.Linear(input_dim, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with learnable scaling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Scaled embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # Linear projection
        embedded = self.projection(x)
        
        # Apply learnable scaling
        embedded = embedded * self.scale
        
        # Layer normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class AdaptiveInputEmbedding(nn.Module):
    """
    Adaptive input embedding that adjusts based on input statistics.
    
    This approach normalizes inputs based on running statistics and
    applies adaptive transformations for better representation learning.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        momentum: Momentum for running statistics (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-5)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        momentum: float = 0.1,
        eps: float = 1e-5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.momentum = momentum
        self.eps = eps
        
        # Running statistics for adaptive normalization
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Learnable parameters for adaptive normalization
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
        # Linear projection
        self.projection = nn.Linear(input_dim, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)
    
    def _update_running_stats(self, x: torch.Tensor):
        """Update running statistics during training."""
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=(0, 1))  # Mean across batch and sequence dimensions
            batch_var = x.var(dim=(0, 1), unbiased=False)
            
            # Update running statistics
            n = self.num_batches_tracked
            momentum = self.momentum if n > 0 else 1.0
            
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_batches_tracked += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Adaptively embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # Update running statistics
        self._update_running_stats(x)
        
        # Adaptive normalization
        if self.training:
            mean = x.mean(dim=(0, 1), keepdim=True)
            var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
        else:
            mean = self.running_mean.view(1, 1, -1)
            var = self.running_var.view(1, 1, -1)
        
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized = normalized * self.weight + self.bias
        
        # Linear projection
        embedded = self.projection(normalized)
        
        # Layer normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class MultiScaleEmbedding(nn.Module):
    """
    Multi-scale embedding that captures features at different temporal scales.
    
    This approach uses multiple convolutional layers with different kernel sizes
    to capture both short-term and long-term patterns before projection.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        kernel_sizes: List of kernel sizes for multi-scale convolution
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        kernel_sizes: List[int] = [1, 3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                input_dim, 
                d_model // len(kernel_sizes), 
                kernel_size=k, 
                padding=k//2
            )
            for k in kernel_sizes
        ])
        
        # Final projection to ensure correct dimension
        total_conv_dim = (d_model // len(kernel_sizes)) * len(kernel_sizes)
        if total_conv_dim != d_model:
            self.final_projection = nn.Linear(total_conv_dim, d_model)
        else:
            self.final_projection = nn.Identity()
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolutional weights."""
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        if isinstance(self.final_projection, nn.Linear):
            nn.init.xavier_uniform_(self.final_projection.weight)
            if self.final_projection.bias is not None:
                nn.init.constant_(self.final_projection.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale embedding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Multi-scale embedded tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Transpose for convolution: (batch_size, input_dim, seq_len)
        x_transposed = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x_transposed))  # Apply ReLU activation
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        concatenated = torch.cat(conv_outputs, dim=1)  # Concat along channel dimension
        
        # Transpose back: (batch_size, seq_len, total_conv_dim)
        concatenated = concatenated.transpose(1, 2)
        
        # Final projection if needed
        embedded = self.final_projection(concatenated)
        
        # Layer normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


def create_input_embedding(
    embedding_type: str = "basic",
    input_dim: int = 7,
    d_model: int = 256,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of input embeddings.
    
    Args:
        embedding_type: Type of embedding ("basic", "feature_wise", "scaled", "adaptive", "multi_scale")
        input_dim: Input feature dimension
        d_model: Model dimension
        dropout: Dropout probability
        **kwargs: Additional arguments for specific embedding types
        
    Returns:
        Input embedding module
    """
    if embedding_type == "basic":
        use_layer_norm = kwargs.get("use_layer_norm", True)
        return InputEmbedding(input_dim, d_model, dropout, use_layer_norm)
    
    elif embedding_type == "feature_wise":
        feature_names = kwargs.get("feature_names", None)
        return FeatureWiseEmbedding(input_dim, d_model, feature_names, dropout)
    
    elif embedding_type == "scaled":
        scale_factor = kwargs.get("scale_factor", None)
        return ScaledInputEmbedding(input_dim, d_model, scale_factor, dropout)
    
    elif embedding_type == "adaptive":
        momentum = kwargs.get("momentum", 0.1)
        eps = kwargs.get("eps", 1e-5)
        return AdaptiveInputEmbedding(input_dim, d_model, momentum, eps, dropout)
    
    elif embedding_type == "multi_scale":
        kernel_sizes = kwargs.get("kernel_sizes", [1, 3, 5, 7])
        return MultiScaleEmbedding(input_dim, d_model, kernel_sizes, dropout)
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")