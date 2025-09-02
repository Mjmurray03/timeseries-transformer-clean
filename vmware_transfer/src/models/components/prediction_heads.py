"""
Prediction Heads Implementation

This module implements various prediction heads for multi-task learning in
time-series forecasting, including price prediction, volatility estimation,
and quantile regression for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class PricePredictionHead(nn.Module):
    """
    Price prediction head for forecasting future stock prices.
    
    Uses a multi-layer network with residual connections to predict
    price changes or absolute prices for the forecast horizon.
    
    Args:
        d_model: Input feature dimension (default: 256)
        forecast_horizon: Number of future time steps to predict (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of hidden layers (default: 2)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function ("relu", "gelu", "tanh") (default: "gelu")
        predict_changes: Whether to predict price changes instead of absolute prices (default: True)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        forecast_horizon: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        predict_changes: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.predict_changes = predict_changes
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(d_model, hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout, self.activation))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, forecast_horizon))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through price prediction head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Price predictions of shape (batch_size, forecast_horizon)
        """
        # Handle sequence input by taking the last time step
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last time step
        
        predictions = self.network(x)
        
        return predictions


class VolatilityPredictionHead(nn.Module):
    """
    Volatility prediction head for estimating future price volatility.
    
    Uses a specialized architecture with positive output constraints
    to ensure volatility predictions are always non-negative.
    
    Args:
        d_model: Input feature dimension (default: 256)
        forecast_horizon: Number of future time steps to predict (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        output_activation: Output activation ("softplus", "exp", "relu") (default: "softplus")
    """
    
    def __init__(
        self,
        d_model: int = 256,
        forecast_horizon: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "softplus"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Output activation to ensure positive volatility
        if output_activation == "softplus":
            self.output_activation = nn.Softplus()
        elif output_activation == "exp":
            self.output_activation = torch.exp
        elif output_activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, -1.0)  # Negative bias for softplus
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through volatility prediction head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Volatility predictions of shape (batch_size, forecast_horizon)
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]
        
        logits = self.network(x)
        volatility = self.output_activation(logits)
        
        return volatility


class QuantileRegressionHead(nn.Module):
    """
    Quantile regression head for uncertainty quantification.
    
    Predicts multiple quantiles (e.g., 10%, 25%, 50%, 75%, 90%) to provide
    confidence intervals around price predictions.
    
    Args:
        d_model: Input feature dimension (default: 256)
        forecast_horizon: Number of future time steps to predict (default: 5)
        quantiles: List of quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        shared_layers: Number of shared layers before quantile-specific heads (default: 2)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        forecast_horizon: int = 5,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        shared_layers: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Shared feature extraction layers
        shared_layers_list = []
        current_dim = d_model
        
        for i in range(shared_layers):
            next_dim = hidden_dim if i == 0 else hidden_dim
            shared_layers_list.extend([
                nn.Linear(current_dim, next_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        self.shared_network = nn.Sequential(*shared_layers_list)
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, forecast_horizon)
            )
            for _ in quantiles
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all quantile heads."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantile regression head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Quantile predictions of shape (batch_size, forecast_horizon, num_quantiles)
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]
        
        # Shared feature extraction
        shared_features = self.shared_network(x)
        
        # Quantile-specific predictions
        quantile_predictions = []
        for head in self.quantile_heads:
            pred = head(shared_features)
            quantile_predictions.append(pred)
        
        # Stack quantiles: (batch_size, forecast_horizon, num_quantiles)
        quantile_predictions = torch.stack(quantile_predictions, dim=-1)
        
        # Ensure quantile ordering (monotonicity constraint)
        quantile_predictions = self._enforce_quantile_ordering(quantile_predictions)
        
        return quantile_predictions
    
    def _enforce_quantile_ordering(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Enforce monotonic ordering of quantile predictions.
        
        Args:
            predictions: Quantile predictions of shape (batch_size, forecast_horizon, num_quantiles)
            
        Returns:
            Monotonically ordered quantile predictions
        """
        # Sort along quantile dimension to ensure ordering
        sorted_predictions, _ = torch.sort(predictions, dim=-1)
        return sorted_predictions


class DirectionalPredictionHead(nn.Module):
    """
    Directional prediction head for binary up/down classification.
    
    Predicts the probability that prices will increase for each time step
    in the forecast horizon.
    
    Args:
        d_model: Input feature dimension (default: 256)
        forecast_horizon: Number of future time steps to predict (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        forecast_horizon: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Network for directional prediction
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Sigmoid()  # Output probabilities
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through directional prediction head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Directional probabilities of shape (batch_size, forecast_horizon)
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]
        
        probabilities = self.network(x)
        
        return probabilities


class ResidualBlock(nn.Module):
    """
    Residual block for deeper prediction heads.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        dropout: Dropout probability
        activation: Activation function
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: nn.Module
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class MultiTaskPredictionHead(nn.Module):
    """
    Multi-task prediction head that combines all prediction tasks.
    
    This head jointly predicts prices, volatility, quantiles, and direction
    with shared feature extraction and task-specific heads.
    
    Args:
        d_model: Input feature dimension (default: 256)
        forecast_horizon: Number of future time steps to predict (default: 5)
        quantiles: List of quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        hidden_dim: Hidden layer dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        task_weights: Weights for different tasks (default: equal weights)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        forecast_horizon: int = 5,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Default task weights
        if task_weights is None:
            task_weights = {
                "price": 1.0,
                "volatility": 1.0,
                "quantiles": 1.0,
                "direction": 1.0
            }
        self.task_weights = task_weights
        
        # Shared feature extraction
        self.shared_network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.price_head = PricePredictionHead(
            hidden_dim, forecast_horizon, hidden_dim // 2, 1, dropout
        )
        
        self.volatility_head = VolatilityPredictionHead(
            hidden_dim, forecast_horizon, hidden_dim // 2, dropout
        )
        
        self.quantile_head = QuantileRegressionHead(
            hidden_dim, forecast_horizon, quantiles, hidden_dim // 2, dropout, 1
        )
        
        self.direction_head = DirectionalPredictionHead(
            hidden_dim, forecast_horizon, hidden_dim // 2, dropout
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize shared network weights."""
        for module in self.shared_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task prediction head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model) or (batch_size, seq_len, d_model)
            
        Returns:
            Dictionary containing predictions for all tasks:
            - "price": Price predictions (batch_size, forecast_horizon)
            - "volatility": Volatility predictions (batch_size, forecast_horizon)
            - "quantiles": Quantile predictions (batch_size, forecast_horizon, num_quantiles)
            - "direction": Directional probabilities (batch_size, forecast_horizon)
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]
        
        # Shared feature extraction
        shared_features = self.shared_network(x)
        
        # Task-specific predictions
        predictions = {
            "price": self.price_head(shared_features),
            "volatility": self.volatility_head(shared_features),
            "quantiles": self.quantile_head(shared_features),
            "direction": self.direction_head(shared_features)
        }
        
        return predictions


# Alias for backward compatibility
PredictionHeads = MultiTaskPredictionHead


def create_prediction_head(
    head_type: str = "multi_task",
    d_model: int = 256,
    forecast_horizon: int = 5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different types of prediction heads.
    
    Args:
        head_type: Type of head ("price", "volatility", "quantiles", "direction", "multi_task")
        d_model: Input feature dimension
        forecast_horizon: Number of future time steps to predict
        **kwargs: Additional arguments for specific head types
        
    Returns:
        Prediction head module
    """
    if head_type == "price":
        return PricePredictionHead(d_model, forecast_horizon, **kwargs)
    
    elif head_type == "volatility":
        return VolatilityPredictionHead(d_model, forecast_horizon, **kwargs)
    
    elif head_type == "quantiles":
        quantiles = kwargs.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
        return QuantileRegressionHead(d_model, forecast_horizon, quantiles, **kwargs)
    
    elif head_type == "direction":
        return DirectionalPredictionHead(d_model, forecast_horizon, **kwargs)
    
    elif head_type == "multi_task":
        return MultiTaskPredictionHead(d_model, forecast_horizon, **kwargs)
    
    else:
        raise ValueError(f"Unknown head type: {head_type}")