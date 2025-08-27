"""Training configuration schema and validation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, validator, Field


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"
    warmup_steps: int = 1000
    max_steps: int = 10000
    min_lr: float = 1e-6
    patience: int = 10
    factor: float = 0.5


@dataclass
class LossConfig:
    """Loss function configuration."""
    price_loss_weight: float = 1.0
    direction_loss_weight: float = 0.5
    volatility_loss_weight: float = 0.3
    quantile_loss_weight: float = 0.2
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Basic training parameters
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Device configuration
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "models/checkpoints"
    save_best_only: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Experiment tracking
    experiment_name: str = "transformer_training"
    project_name: str = "timeseries-transformer"
    log_every: int = 100
    
    # Validation
    val_every: int = 1
    val_metric: str = "loss"
    
    # Component configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Data configuration
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        # Check splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Check positive values
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        # Check learning rate
        if self.optimizer.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Check quantiles
        if not all(0 < q < 1 for q in self.loss.quantiles):
            raise ValueError("All quantiles must be between 0 and 1")
        
        # Check device
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device: {self.device}")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        optimizer_config = config_dict.pop('optimizer', {})
        scheduler_config = config_dict.pop('scheduler', {})
        loss_config = config_dict.pop('loss', {})
        
        # Create nested config objects
        optimizer = OptimizerConfig(**optimizer_config)
        scheduler = SchedulerConfig(**scheduler_config)
        loss = LossConfig(**loss_config)
        
        # Create main config
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            **config_dict
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, (OptimizerConfig, SchedulerConfig, LossConfig)):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        return config_dict
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


class TrainingConfigValidator(BaseModel):
    """Pydantic validator for training configuration."""
    
    num_epochs: int = Field(gt=0, description="Number of training epochs")
    batch_size: int = Field(gt=0, description="Batch size for training")
    learning_rate: float = Field(gt=0, le=1, description="Learning rate")
    gradient_clip: float = Field(gt=0, description="Gradient clipping threshold")
    
    train_split: float = Field(gt=0, lt=1, description="Training data split")
    val_split: float = Field(gt=0, lt=1, description="Validation data split")
    test_split: float = Field(gt=0, lt=1, description="Test data split")
    
    @validator('train_split', 'val_split', 'test_split')
    def validate_splits(cls, v, values):
        """Validate that splits sum to 1.0."""
        if 'train_split' in values and 'val_split' in values:
            total = values['train_split'] + values['val_split'] + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Splits must sum to 1.0, got {total}")
        return v
    
    class Config:
        extra = "allow"  # Allow additional fields


def create_default_config() -> TrainingConfig:
    """Create default training configuration."""
    return TrainingConfig()


def create_quick_test_config() -> TrainingConfig:
    """Create configuration for quick testing."""
    return TrainingConfig(
        num_epochs=5,
        batch_size=16,
        log_every=10,
        save_every=2,
        early_stopping_patience=3,
        experiment_name="quick_test"
    )


def create_production_config() -> TrainingConfig:
    """Create configuration for production training."""
    return TrainingConfig(
        num_epochs=200,
        batch_size=64,
        gradient_accumulation_steps=2,
        optimizer=OptimizerConfig(
            learning_rate=5e-5,
            weight_decay=0.01
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            warmup_steps=2000,
            max_steps=50000
        ),
        early_stopping_patience=20,
        experiment_name="production_training"
    )