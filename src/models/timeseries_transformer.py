"""
Time-Series Transformer Model Implementation

This module implements the main TimeSeriesTransformer model that combines all
components for stock price forecasting with multi-task learning capabilities.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.attention_pooling import AttentionPooling
from .components.input_embedding import InputEmbedding
from .components.positional_encoding import LearnedPositionalEncoding
from .components.prediction_heads import MultiTaskPredictionHead
from .components.transformer_block import TransformerBlock


class TimeSeriesTransformer(nn.Module):
    """
    Time-Series Transformer for stock price forecasting.

    This model follows the architecture specified in the design documents and
    implements multi-task learning for price, volatility, quantiles, and direction.

    Architecture:
    1. Input Embedding (Linear projection + Layer Norm)
    2. Positional Encoding (Learnable parameters)
    3. Transformer Blocks (Multi-head attention + FFN)
    4. Attention Pooling (Aggregate sequence representation)
    5. Multi-Task Prediction Heads (Price, Volatility, Quantiles, Direction)

    Args:
        input_dim: Input feature dimension (default: 10 for OHLCV + 5 indicators)
        hidden_dim: Model dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout probability (default: 0.1)
        max_seq_length: Maximum sequence length (default: 60)
        output_dim: Output dimension (default: 3 for [price, direction, volatility])
        forecast_horizon: Number of future time steps to predict (default: 5)
        quantiles: List of quantile levels for uncertainty quantification
        use_attention_pooling: Whether to use attention pooling (default: True)
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 60,
        output_dim: int = 3,
        forecast_horizon: int = 5,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        use_attention_pooling: bool = True,
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
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.use_attention_pooling = use_attention_pooling

        # Validation
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        assert max_seq_length > 0, f"max_seq_length must be positive, got {max_seq_length}"
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"

        # 1. Input Embedding
        self.input_embedding = InputEmbedding(
            input_dim=input_dim, d_model=hidden_dim, dropout=dropout, use_layer_norm=True
        )

        # 2. Positional Encoding
        self.positional_encoding = LearnedPositionalEncoding(
            max_seq_len=max_seq_length, d_model=hidden_dim, dropout=dropout
        )

        # 3. Transformer Blocks
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=hidden_dim,
                    n_heads=num_heads,
                    d_ff=hidden_dim * 4,  # Standard FFN expansion
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Attention Pooling (Optional)
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(
                d_model=hidden_dim, dropout=dropout, temperature=1.0
            )
        else:
            self.attention_pooling = None

        # 5. Output Layer - Simple linear for the test specification
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Alternative: Multi-Task Heads (can be enabled later)
        self.multi_task_heads = MultiTaskPredictionHead(
            d_model=hidden_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all model weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        use_multi_task: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the time-series transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights from all layers
            use_multi_task: Whether to use multi-task heads instead of simple output

        Returns:
            If return_attention=False:
                output: Predictions of shape (batch_size, output_dim)
            If return_attention=True:
                output: Predictions of shape (batch_size, output_dim)
                attention_weights: List of attention weights from each layer
        """
        batch_size, seq_len, input_dim = x.shape

        # Validation
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        assert (
            seq_len <= self.max_seq_length
        ), f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}"

        # Store attention weights if requested
        attention_weights = [] if return_attention else None

        # 1. Input Embedding
        embedded = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)

        # 2. Positional Encoding
        encoded = self.positional_encoding(embedded)  # (batch_size, seq_len, hidden_dim)

        # 3. Transformer Layers
        hidden_states = encoded
        for i, layer in enumerate(self.transformer_layers):
            if return_attention:
                hidden_states, layer_attention = layer(
                    hidden_states, mask=mask, return_attention=True
                )
                attention_weights.append(layer_attention)
            else:
                hidden_states = layer(hidden_states, mask=mask, return_attention=False)

        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 4. Sequence Aggregation
        if self.use_attention_pooling:
            if return_attention:
                pooled_representation, pooling_attention = self.attention_pooling(
                    hidden_states, mask=mask, return_attention=True
                )
                # Add pooling attention to the list (optional)
            else:
                pooled_representation = self.attention_pooling(
                    hidden_states, mask=mask, return_attention=False
                )
        else:
            # Simple mean pooling with mask consideration
            if mask is not None:
                # Apply mask and compute weighted average
                mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
                masked_hidden = hidden_states * mask_expanded
                pooled_representation = masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                # Simple average pooling
                pooled_representation = hidden_states.mean(dim=1)

        # 5. Output Prediction
        if use_multi_task:
            # Use multi-task heads
            output = self.multi_task_heads(pooled_representation)
        else:
            # Simple linear output for basic testing
            output = self.output_layer(pooled_representation)  # (batch_size, output_dim)

        # Return results
        if return_attention:
            return output, attention_weights
        else:
            return output

    def get_attention_weights(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, layer_idx: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract attention weights from specified layer(s).

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            layer_idx: Specific layer index (None for all layers)

        Returns:
            Attention weights from specified layer(s)
        """
        _, attention_weights = self.forward(x, mask, return_attention=True)

        if layer_idx is not None:
            assert 0 <= layer_idx < len(attention_weights), f"Layer index {layer_idx} out of range"
            return attention_weights[layer_idx]
        else:
            return attention_weights

    def predict_multi_task(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make multi-task predictions.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dictionary containing predictions for all tasks
        """
        return self.forward(x, mask, return_attention=False, use_multi_task=True)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        component_params = {}
        component_params["input_embedding"] = sum(
            p.numel() for p in self.input_embedding.parameters()
        )
        component_params["positional_encoding"] = sum(
            p.numel() for p in self.positional_encoding.parameters()
        )
        component_params["transformer_layers"] = sum(
            p.numel() for p in self.transformer_layers.parameters()
        )
        if self.attention_pooling:
            component_params["attention_pooling"] = sum(
                p.numel() for p in self.attention_pooling.parameters()
            )
        component_params["output_layer"] = sum(p.numel() for p in self.output_layer.parameters())
        component_params["multi_task_heads"] = sum(
            p.numel() for p in self.multi_task_heads.parameters()
        )

        return {
            "total": total_params,
            "trainable": trainable_params,
            "components": component_params,
        }

    def get_model_info(self) -> Dict[str, Union[int, float, List]]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model configuration and statistics
        """
        param_counts = self.count_parameters()

        return {
            "architecture": "TimeSeriesTransformer",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
            "output_dim": self.output_dim,
            "forecast_horizon": self.forecast_horizon,
            "quantiles": self.quantiles,
            "dropout": self.dropout,
            "use_attention_pooling": self.use_attention_pooling,
            "total_parameters": param_counts["total"],
            "trainable_parameters": param_counts["trainable"],
            "parameter_breakdown": param_counts["components"],
        }
