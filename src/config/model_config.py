"""
Model configuration classes and validation.
Provides typed access to model architecture configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model architecture."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 60
    forecast_horizon: int = 5
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        """Validate transformer configuration."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class LSTMConfig:
    """Configuration for LSTM baseline model."""

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    batch_first: bool = True


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture components."""

    input_features: int = 7  # OHLCV + 2 technical indicators
    output_features: int = 1  # Close price prediction
    hidden_dim: int = 256
    num_layers: int = 6
    attention_heads: int = 8
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True
    positional_encoding_type: str = "sinusoidal"  # sinusoidal, learned


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    primary_loss: str = "mse"  # mse, mae, huber, quantile
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"regression": 0.7, "direction": 0.3}
    )
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    huber_delta: float = 1.0


@dataclass
class RegularizationConfig:
    """Configuration for regularization techniques."""

    weight_decay: float = 1e-4
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0  # 0.0 disables mixup


class ModelConfig:
    """
    Main model configuration class.
    Provides typed access to all model architecture configuration.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize model configuration from dictionary.

        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self.raw_config = config_dict
        self.model_type = config_dict.get("model_type", "transformer")

        # Parse architecture-specific configurations
        if self.model_type == "transformer":
            transformer_config = config_dict.get("transformer", {})
            self.transformer = TransformerConfig(**transformer_config)
        elif self.model_type == "lstm":
            lstm_config = config_dict.get("lstm", {})
            self.lstm = LSTMConfig(**lstm_config)

        # Parse general model configuration
        arch_config = config_dict.get("architecture", {})
        self.architecture = ModelArchitectureConfig(**arch_config)

        # Parse loss configuration
        loss_config = config_dict.get("loss", {})
        self.loss = LossConfig(**loss_config)

        # Parse regularization configuration
        reg_config = config_dict.get("regularization", {})
        self.regularization = RegularizationConfig(**reg_config)

        # Store other configurations
        self.optimization = config_dict.get("optimization", {})
        self.training = config_dict.get("training", {})

    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters for initialization."""
        params = {
            "input_features": self.architecture.input_features,
            "output_features": self.architecture.output_features,
            "hidden_dim": self.architecture.hidden_dim,
            "dropout_rate": self.architecture.dropout_rate,
        }

        if self.model_type == "transformer":
            params.update(
                {
                    "d_model": self.transformer.d_model,
                    "n_heads": self.transformer.n_heads,
                    "n_layers": self.transformer.n_layers,
                    "max_seq_length": self.transformer.max_seq_length,
                    "forecast_horizon": self.transformer.forecast_horizon,
                }
            )
        elif self.model_type == "lstm":
            params.update(
                {
                    "hidden_size": self.lstm.hidden_size,
                    "num_layers": self.lstm.num_layers,
                    "bidirectional": self.lstm.bidirectional,
                }
            )

        return params

    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss function configuration."""
        return {
            "primary_loss": self.loss.primary_loss,
            "loss_weights": self.loss.loss_weights,
            "quantile_levels": self.loss.quantile_levels,
            "huber_delta": self.loss.huber_delta,
        }

    def get_regularization_config(self) -> Dict[str, Any]:
        """Get regularization configuration."""
        return {
            "weight_decay": self.regularization.weight_decay,
            "gradient_clipping": self.regularization.gradient_clipping,
            "max_grad_norm": self.regularization.max_grad_norm,
            "label_smoothing": self.regularization.label_smoothing,
            "mixup_alpha": self.regularization.mixup_alpha,
        }

    def validate(self) -> bool:
        """
        Validate model configuration for consistency.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model type
            if self.model_type not in ["transformer", "lstm", "ensemble"]:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False

            # Validate architecture parameters
            if self.architecture.input_features <= 0:
                logger.error("input_features must be positive")
                return False

            if self.architecture.output_features <= 0:
                logger.error("output_features must be positive")
                return False

            # Validate loss configuration
            if self.loss.primary_loss not in ["mse", "mae", "huber", "quantile"]:
                logger.error(f"Unsupported loss function: {self.loss.primary_loss}")
                return False

            # Validate quantile levels
            for level in self.loss.quantile_levels:
                if not 0 < level < 1:
                    logger.error(f"Quantile level must be between 0 and 1, got {level}")
                    return False

            # Validate regularization parameters
            if not 0 <= self.regularization.weight_decay <= 1:
                logger.error("weight_decay must be between 0 and 1")
                return False

            if self.regularization.max_grad_norm <= 0:
                logger.error("max_grad_norm must be positive")
                return False

            logger.info("Model configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Model configuration validation error: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration back to dictionary format."""
        return self.raw_config
