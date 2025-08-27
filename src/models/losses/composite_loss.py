"""
Composite loss function for time-series transformer model.

This module implements a multi-objective loss function that combines:
- Price prediction loss (MSE)
- Direction classification loss (Cross-entropy)
- Volatility prediction loss (MSE)
- Quantile regression loss (Pinball loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .quantile_loss import QuantileLoss
from .directional_loss import DirectionalLoss


class CompositeLoss(nn.Module):
    """
    Multi-objective loss function combining price, direction, volatility, and quantile losses.
    
    Loss components:
    - Price loss (50%): MSE between predicted and actual prices
    - Direction loss (30%): Cross-entropy for price movement direction
    - Volatility loss (10%): MSE between predicted and actual volatility
    - Quantile loss (10%): Pinball loss for uncertainty quantification
    """
    
    def __init__(
        self,
        price_weight: float = 0.5,
        direction_weight: float = 0.3,
        volatility_weight: float = 0.1,
        quantile_weight: float = 0.1,
        quantile_levels: Optional[list] = None
    ):
        """
        Initialize composite loss function.
        
        Args:
            price_weight: Weight for price prediction loss
            direction_weight: Weight for direction classification loss
            volatility_weight: Weight for volatility prediction loss
            quantile_weight: Weight for quantile regression loss
            quantile_levels: List of quantile levels for uncertainty estimation
        """
        super().__init__()
        
        # Validate weights sum to 1.0
        total_weight = price_weight + direction_weight + volatility_weight + quantile_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.quantile_weight = quantile_weight
        
        # Default quantile levels for uncertainty estimation
        if quantile_levels is None:
            quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantile_levels = quantile_levels
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.directional_loss = DirectionalLoss(loss_type='bce')
        self.quantile_loss = QuantileLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate composite loss from model predictions and targets.
        
        Args:
            predictions: Dictionary containing model predictions:
                - 'price': Price predictions [batch_size, horizon]
                - 'volatility': Volatility predictions [batch_size, horizon]
                - 'quantiles': Quantile predictions [batch_size, horizon, n_quantiles]
            targets: Dictionary containing target values:
                - 'price': True prices [batch_size, horizon]
                - 'volatility': True volatility [batch_size, horizon]
                
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        batch_size, horizon = predictions['price'].shape
        
        # 1. Price prediction loss (MSE)
        price_loss = self.mse_loss(predictions['price'], targets['price'])
        
        # 2. Direction classification loss
        direction_loss = self.directional_loss(predictions['price'], targets['price'])
        
        # 3. Volatility prediction loss (MSE)
        volatility_loss = self.mse_loss(predictions['volatility'], targets['volatility'])
        
        # 4. Quantile regression loss
        quantile_loss_total = 0.0
        n_quantiles = len(self.quantile_levels)
        
        for i, quantile_level in enumerate(self.quantile_levels):
            q_pred = predictions['quantiles'][:, :, i]  # [batch_size, horizon]
            q_loss = self.quantile_loss(q_pred, targets['price'], quantile_level)
            quantile_loss_total += q_loss
        
        # Average over quantiles
        quantile_loss_avg = quantile_loss_total / n_quantiles
        
        # 5. Combine losses with weights
        total_loss = (
            self.price_weight * price_loss +
            self.direction_weight * direction_loss +
            self.volatility_weight * volatility_loss +
            self.quantile_weight * quantile_loss_avg
        )
        
        # Return loss components for logging
        loss_components = {
            'price_loss': price_loss.item(),
            'direction_loss': direction_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'quantile_loss': quantile_loss_avg.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def get_weights(self) -> Dict[str, float]:
        """Return current loss weights."""
        return {
            'price_weight': self.price_weight,
            'direction_weight': self.direction_weight,
            'volatility_weight': self.volatility_weight,
            'quantile_weight': self.quantile_weight
        }
    
    def update_weights(self, **kwargs):
        """Update loss weights dynamically during training."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown weight parameter: {key}")
        
        # Validate weights still sum to 1.0
        total_weight = (
            self.price_weight + self.direction_weight + 
            self.volatility_weight + self.quantile_weight
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Updated weights must sum to 1.0, got {total_weight}")