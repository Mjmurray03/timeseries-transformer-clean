"""
Directional loss implementation for time-series prediction.

This module provides loss functions specifically designed for predicting
the direction of price movements (up/down classification).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Loss function for predicting price movement direction.

    This loss treats direction prediction as a binary classification problem,
    where the model predicts whether the next price will be higher or lower
    than the current price.
    """

    def __init__(self, loss_type: str = "bce", class_weights: Optional[torch.Tensor] = None):
        """
        Initialize directional loss.

        Args:
            loss_type: Type of loss function to use:
                      'bce' - Binary Cross Entropy
                      'focal' - Focal Loss (for imbalanced data)
                      'hinge' - Hinge Loss (SVM-style)
            class_weights: Optional weights for classes [down_weight, up_weight]
        """
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=class_weights)
        elif loss_type == "hinge":
            self.loss_fn = HingeLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, price_predictions: torch.Tensor, price_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate directional loss from price predictions.

        Args:
            price_predictions: Predicted prices [batch_size, horizon]
            price_targets: True prices [batch_size, horizon]

        Returns:
            Directional classification loss
        """
        # Calculate returns (price changes)
        pred_returns = torch.diff(price_predictions, dim=1)  # [batch, horizon-1]
        true_returns = torch.diff(price_targets, dim=1)  # [batch, horizon-1]

        # Convert to binary labels (1 for up, 0 for down)
        true_direction = (true_returns > 0).float()

        # Use predicted returns as logits for classification
        pred_logits = pred_returns

        return self.loss_fn(pred_logits, true_direction)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in directional prediction.

    Focal Loss = -α(1-p_t)^γ * log(p_t)

    Where p_t is the model's estimated probability for the true class.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class (optional)
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs: Predicted logits [batch_size, ...]
            targets: True binary labels [batch_size, ...]

        Returns:
            Focal loss tensor
        """
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate probabilities
        p_t = torch.exp(-bce_loss)

        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * bce_loss

        return focal_loss.mean()


class HingeLoss(nn.Module):
    """
    Hinge Loss for directional prediction (SVM-style loss).

    Hinge Loss = max(0, 1 - y * f(x))

    Where y ∈ {-1, +1} and f(x) is the prediction.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize Hinge Loss.

        Args:
            margin: Margin parameter for the hinge loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate hinge loss.

        Args:
            inputs: Predicted values [batch_size, ...]
            targets: True binary labels in {0, 1} [batch_size, ...]

        Returns:
            Hinge loss tensor
        """
        # Convert targets from {0, 1} to {-1, +1}
        y = 2 * targets - 1

        # Calculate hinge loss
        loss = torch.clamp(self.margin - y * inputs, min=0)

        return loss.mean()


class MultiHorizonDirectionalLoss(nn.Module):
    """
    Directional loss for multi-horizon predictions with different weights.

    This allows giving different importance to short-term vs long-term
    directional accuracy.
    """

    def __init__(self, horizon_weights: Optional[torch.Tensor] = None, loss_type: str = "bce"):
        """
        Initialize multi-horizon directional loss.

        Args:
            horizon_weights: Weights for different prediction horizons
            loss_type: Base loss function type
        """
        super().__init__()
        self.horizon_weights = horizon_weights
        self.directional_loss = DirectionalLoss(loss_type=loss_type)

    def forward(self, price_predictions: torch.Tensor, price_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted multi-horizon directional loss.

        Args:
            price_predictions: Predicted prices [batch_size, horizon]
            price_targets: True prices [batch_size, horizon]

        Returns:
            Weighted directional loss across horizons
        """
        batch_size, horizon = price_predictions.shape

        if self.horizon_weights is None:
            # Equal weights for all horizons
            self.horizon_weights = torch.ones(horizon - 1)

        if self.horizon_weights.device != price_predictions.device:
            self.horizon_weights = self.horizon_weights.to(price_predictions.device)

        total_loss = 0.0

        # Calculate loss for each prediction step
        for i in range(horizon - 1):
            # Get predictions and targets for this step
            step_pred = price_predictions[:, : i + 2]  # [batch, i+2]
            step_target = price_targets[:, : i + 2]  # [batch, i+2]

            # Calculate directional loss for this step
            step_loss = self.directional_loss(step_pred, step_target)

            # Weight by horizon importance
            weighted_loss = self.horizon_weights[i] * step_loss
            total_loss += weighted_loss

        # Normalize by sum of weights
        return total_loss / self.horizon_weights.sum()


class BalancedDirectionalLoss(nn.Module):
    """
    Directional loss with automatic class balancing.

    This loss automatically adjusts for class imbalance in the training data
    by computing class frequencies and adjusting weights accordingly.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize balanced directional loss.

        Args:
            smoothing: Smoothing factor for class weight updates
        """
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("class_counts", torch.zeros(2))
        self.register_buffer("total_samples", torch.tensor(0.0))

    def forward(self, price_predictions: torch.Tensor, price_targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate balanced directional loss.

        Args:
            price_predictions: Predicted prices [batch_size, horizon]
            price_targets: True prices [batch_size, horizon]

        Returns:
            Balanced directional loss
        """
        # Calculate returns and directions
        pred_returns = torch.diff(price_predictions, dim=1)
        true_returns = torch.diff(price_targets, dim=1)
        true_direction = (true_returns > 0).float()

        # Update class counts (exponential moving average)
        with torch.no_grad():
            batch_up = true_direction.sum()
            batch_down = (1 - true_direction).sum()
            batch_total = true_direction.numel()

            # Update counts with smoothing
            self.class_counts[0] = (1 - self.smoothing) * self.class_counts[
                0
            ] + self.smoothing * batch_down
            self.class_counts[1] = (1 - self.smoothing) * self.class_counts[
                1
            ] + self.smoothing * batch_up
            self.total_samples = (
                1 - self.smoothing
            ) * self.total_samples + self.smoothing * batch_total

        # Calculate class weights (inverse frequency)
        class_frequencies = self.class_counts / (self.total_samples + 1e-8)
        class_weights = 1.0 / (class_frequencies + 1e-8)
        class_weights = class_weights / class_weights.sum()  # Normalize

        # Use weighted BCE loss
        pos_weight = class_weights[1] / class_weights[0]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return loss_fn(pred_returns, true_direction)
