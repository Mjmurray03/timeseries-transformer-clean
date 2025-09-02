"""
Quantile regression loss implementation.

This module provides the quantile loss (also known as pinball loss) for
uncertainty quantification in time-series predictions.
"""

from typing import Union

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    Quantile regression loss (Pinball loss) for uncertainty estimation.

    The quantile loss is asymmetric and penalizes over-prediction and under-prediction
    differently based on the desired quantile level.

    For quantile τ ∈ (0,1):
    L_τ(y, ŷ) = max(τ(y - ŷ), (τ - 1)(y - ŷ))

    Where:
    - y is the true value
    - ŷ is the predicted quantile
    - τ is the quantile level (e.g., 0.1 for 10th percentile)
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize quantile loss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'
        """
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, quantile: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate quantile loss.

        Args:
            predictions: Predicted quantile values [..., *]
            targets: True target values [..., *] (same shape as predictions)
            quantile: Quantile level(s) in (0, 1). Can be:
                     - float: Single quantile level
                     - tensor: Multiple quantile levels (must broadcast with predictions)

        Returns:
            Quantile loss tensor
        """
        if isinstance(quantile, (int, float)):
            if not 0 < quantile < 1:
                raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
            quantile = torch.tensor(quantile, device=predictions.device, dtype=predictions.dtype)

        # Calculate prediction errors
        errors = targets - predictions

        # Quantile loss: max(τ * errors, (τ - 1) * errors)
        loss = torch.max(quantile * errors, (quantile - 1) * errors)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class MultiQuantileLoss(nn.Module):
    """
    Multi-quantile loss for simultaneous prediction of multiple quantiles.

    This is useful for models that predict multiple quantiles simultaneously
    to capture prediction uncertainty.
    """

    def __init__(self, quantile_levels: list, reduction: str = "mean"):
        """
        Initialize multi-quantile loss.

        Args:
            quantile_levels: List of quantile levels to compute loss for
            reduction: Reduction method for individual quantile losses
        """
        super().__init__()
        self.quantile_levels = sorted(quantile_levels)
        self.quantile_loss = QuantileLoss(reduction=reduction)

        # Validate quantile levels
        for q in self.quantile_levels:
            if not 0 < q < 1:
                raise ValueError(f"All quantiles must be in (0, 1), got {q}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate multi-quantile loss.

        Args:
            predictions: Predicted quantiles [..., n_quantiles]
            targets: True target values [..., 1] or [...] (will be broadcast)

        Returns:
            Average quantile loss across all quantile levels
        """
        if predictions.shape[-1] != len(self.quantile_levels):
            raise ValueError(
                f"Last dimension of predictions ({predictions.shape[-1]}) "
                f"must match number of quantile levels ({len(self.quantile_levels)})"
            )

        # Ensure targets can be broadcast with predictions
        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-1)

        total_loss = 0.0

        for i, quantile_level in enumerate(self.quantile_levels):
            q_pred = predictions[..., i]
            q_targets = targets[..., 0] if targets.shape[-1] == 1 else targets[..., i]

            q_loss = self.quantile_loss(q_pred, q_targets, quantile_level)
            total_loss += q_loss

        # Return average loss across quantiles
        return total_loss / len(self.quantile_levels)

    def get_quantile_levels(self) -> list:
        """Return the quantile levels used by this loss."""
        return self.quantile_levels.copy()


class AdaptiveQuantileLoss(nn.Module):
    """
    Adaptive quantile loss that adjusts quantile levels during training.

    This can be useful for curriculum learning or when the optimal
    quantile levels are not known a priori.
    """

    def __init__(
        self,
        initial_quantiles: list,
        adaptation_rate: float = 0.01,
        min_quantile: float = 0.01,
        max_quantile: float = 0.99,
    ):
        """
        Initialize adaptive quantile loss.

        Args:
            initial_quantiles: Initial quantile levels
            adaptation_rate: Rate at which quantiles adapt
            min_quantile: Minimum allowed quantile level
            max_quantile: Maximum allowed quantile level
        """
        super().__init__()
        self.adaptation_rate = adaptation_rate
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

        # Store quantiles as learnable parameters
        self.quantile_params = nn.Parameter(torch.tensor(initial_quantiles, dtype=torch.float32))

        self.quantile_loss = QuantileLoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate adaptive quantile loss.

        Args:
            predictions: Predicted quantiles [..., n_quantiles]
            targets: True target values [..., 1] or [...]

        Returns:
            Average quantile loss with adaptive quantile levels
        """
        # Apply sigmoid to ensure quantiles are in (0, 1)
        quantiles = torch.sigmoid(self.quantile_params)
        quantiles = self.min_quantile + (self.max_quantile - self.min_quantile) * quantiles

        # Sort quantiles to maintain order
        quantiles, _ = torch.sort(quantiles)

        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-1)

        total_loss = 0.0

        for i, quantile_level in enumerate(quantiles):
            q_pred = predictions[..., i]
            q_targets = targets[..., 0] if targets.shape[-1] == 1 else targets[..., i]

            q_loss = self.quantile_loss(q_pred, q_targets, quantile_level)
            total_loss += q_loss

        return total_loss / len(quantiles)

    def get_current_quantiles(self) -> torch.Tensor:
        """Get current quantile levels."""
        quantiles = torch.sigmoid(self.quantile_params)
        quantiles = self.min_quantile + (self.max_quantile - self.min_quantile) * quantiles
        return torch.sort(quantiles)[0]
