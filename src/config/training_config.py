"""Training configuration schema and validation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import argparse
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
    use_composite_loss: bool = False  # Disabled for basic price prediction
    price_weight: float = 0.5
    direction_weight: float = 0.3
    volatility_weight: float = 0.1
    quantile_weight: float = 0.1
    
    # Legacy fields for backwards compatibility
    price_loss_weight: float = field(init=False)
    direction_loss_weight: float = field(init=False)
    volatility_loss_weight: float = field(init=False)
    quantile_loss_weight: float = field(init=False)
    
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    
    def __post_init__(self):
        """Set legacy field values for backwards compatibility."""
        self.price_loss_weight = self.price_weight
        self.direction_loss_weight = self.direction_weight
        self.volatility_loss_weight = self.volatility_weight
        self.quantile_loss_weight = self.quantile_weight


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 30  # Number of input features
    hidden_dim: int = 256  # Model dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dropout: float = 0.1  # Dropout probability
    max_seq_length: int = 60  # Maximum sequence length
    output_dim: int = 5  # Output dimension (forecast horizon)
    forecast_horizon: int = 5  # Number of future time steps
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    use_attention_pooling: bool = True  # Use attention pooling
    model_version: str = "1.0"  # Model version for tracking


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
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Data configuration
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    tickers: List[str] = field(default_factory=lambda: ['AAPL'])
    
    # Training steps (calculated dynamically)
    steps_per_epoch: Optional[int] = None
    warmup_steps: int = 0
    
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
    
    @classmethod
    def from_args(cls, args: Union[argparse.Namespace, Dict[str, Any]]) -> "TrainingConfig":
        """
        Create configuration from command-line arguments.
        
        This method maps command-line argument names to TrainingConfig parameters,
        handles missing optional parameters with defaults, validates parameter types
        and ranges, and returns a properly initialized TrainingConfig instance.
        
        Args:
            args: argparse.Namespace from command-line parsing or dictionary
            
        Returns:
            TrainingConfig: Properly initialized configuration instance
            
        Raises:
            ValueError: If required parameters are missing or invalid
            TypeError: If parameter types are incorrect
            
        Parameter Mapping:
            Command-line argument -> TrainingConfig parameter
            --epochs -> num_epochs
            --batch-size -> batch_size
            --learning-rate -> optimizer.learning_rate
            --device -> device
            --experiment-name -> experiment_name
            --checkpoint-dir -> checkpoint_dir
            --seed -> seed
            (and many more - see implementation for complete mapping)
        """
        # Convert argparse.Namespace to dictionary if needed
        if isinstance(args, argparse.Namespace):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args.copy()
        else:
            raise TypeError(f"Expected argparse.Namespace or dict, got {type(args)}")
        
        # Initialize configuration dictionaries
        config_params = {}
        model_params = {}
        optimizer_params = {}
        scheduler_params = {}
        loss_params = {}
        
        # Define parameter mapping: command-line arg -> (config_section, param_name, type_converter, validator_func)
        param_mapping = {
            # Basic training parameters
            'epochs': ('config', 'num_epochs', int, lambda x: cls._validate_positive_int(x, 'num_epochs')),
            'num_epochs': ('config', 'num_epochs', int, lambda x: cls._validate_positive_int(x, 'num_epochs')),
            'batch_size': ('config', 'batch_size', int, lambda x: cls._validate_positive_int(x, 'batch_size')),
            'batch-size': ('config', 'batch_size', int, lambda x: cls._validate_positive_int(x, 'batch_size')),
            'gradient_accumulation_steps': ('config', 'gradient_accumulation_steps', int, lambda x: cls._validate_positive_int(x, 'gradient_accumulation_steps')),
            'gradient-accumulation-steps': ('config', 'gradient_accumulation_steps', int, lambda x: cls._validate_positive_int(x, 'gradient_accumulation_steps')),
            'gradient_clip': ('config', 'gradient_clip', float, lambda x: cls._validate_positive_float(x, 'gradient_clip')),
            'gradient-clip': ('config', 'gradient_clip', float, lambda x: cls._validate_positive_float(x, 'gradient_clip')),
            
            # Mixed precision and device
            'use_amp': ('config', 'use_amp', bool, None),
            'use-amp': ('config', 'use_amp', bool, None),
            'device': ('config', 'device', str, lambda x: cls._validate_device(x)),
            'num_workers': ('config', 'num_workers', int, lambda x: cls._validate_non_negative_int(x, 'num_workers')),
            'num-workers': ('config', 'num_workers', int, lambda x: cls._validate_non_negative_int(x, 'num_workers')),
            'pin_memory': ('config', 'pin_memory', bool, None),
            'pin-memory': ('config', 'pin_memory', bool, None),
            
            # Checkpointing
            'save_every': ('config', 'save_every', int, lambda x: cls._validate_positive_int(x, 'save_every')),
            'save-every': ('config', 'save_every', int, lambda x: cls._validate_positive_int(x, 'save_every')),
            'checkpoint_dir': ('config', 'checkpoint_dir', str, None),
            'checkpoint-dir': ('config', 'checkpoint_dir', str, None),
            'save_best_only': ('config', 'save_best_only', bool, None),
            'save-best-only': ('config', 'save_best_only', bool, None),
            
            # Early stopping
            'early_stopping_patience': ('config', 'early_stopping_patience', int, lambda x: cls._validate_positive_int(x, 'early_stopping_patience')),
            'early-stopping-patience': ('config', 'early_stopping_patience', int, lambda x: cls._validate_positive_int(x, 'early_stopping_patience')),
            'early_stopping_min_delta': ('config', 'early_stopping_min_delta', float, lambda x: cls._validate_non_negative_float(x, 'early_stopping_min_delta')),
            'early-stopping-min-delta': ('config', 'early_stopping_min_delta', float, lambda x: cls._validate_non_negative_float(x, 'early_stopping_min_delta')),
            
            # Experiment tracking
            'experiment_name': ('config', 'experiment_name', str, None),
            'experiment-name': ('config', 'experiment_name', str, None),
            'project_name': ('config', 'project_name', str, None),
            'project-name': ('config', 'project_name', str, None),
            'log_every': ('config', 'log_every', int, lambda x: cls._validate_positive_int(x, 'log_every')),
            'log-every': ('config', 'log_every', int, lambda x: cls._validate_positive_int(x, 'log_every')),
            
            # Validation
            'val_every': ('config', 'val_every', int, lambda x: cls._validate_positive_int(x, 'val_every')),
            'val-every': ('config', 'val_every', int, lambda x: cls._validate_positive_int(x, 'val_every')),
            'val_metric': ('config', 'val_metric', str, None),
            'val-metric': ('config', 'val_metric', str, None),
            
            # Data splits
            'train_split': ('config', 'train_split', float, lambda x: cls._validate_split(x, 'train_split')),
            'train-split': ('config', 'train_split', float, lambda x: cls._validate_split(x, 'train_split')),
            'val_split': ('config', 'val_split', float, lambda x: cls._validate_split(x, 'val_split')),
            'val-split': ('config', 'val_split', float, lambda x: cls._validate_split(x, 'val_split')),
            'test_split': ('config', 'test_split', float, lambda x: cls._validate_split(x, 'test_split')),
            'test-split': ('config', 'test_split', float, lambda x: cls._validate_split(x, 'test_split')),
            
            # Data parameters
            'tickers': ('config', 'tickers', list, None),
            'warmup-steps': ('config', 'warmup_steps', int, lambda x: cls._validate_non_negative_int(x, 'warmup_steps')),
            'warmup_steps': ('config', 'warmup_steps', int, lambda x: cls._validate_non_negative_int(x, 'warmup_steps')),
            
            # Reproducibility
            'seed': ('config', 'seed', int, lambda x: cls._validate_non_negative_int(x, 'seed')),
            'deterministic': ('config', 'deterministic', bool, None),
            
            # Model architecture parameters
            'input-dim': ('model', 'input_dim', int, lambda x: cls._validate_positive_int(x, 'input_dim')),
            'input_dim': ('model', 'input_dim', int, lambda x: cls._validate_positive_int(x, 'input_dim')),
            'hidden-dim': ('model', 'hidden_dim', int, lambda x: cls._validate_positive_int(x, 'hidden_dim')),
            'hidden_dim': ('model', 'hidden_dim', int, lambda x: cls._validate_positive_int(x, 'hidden_dim')),
            'num-heads': ('model', 'num_heads', int, lambda x: cls._validate_positive_int(x, 'num_heads')),
            'num_heads': ('model', 'num_heads', int, lambda x: cls._validate_positive_int(x, 'num_heads')),
            'num-layers': ('model', 'num_layers', int, lambda x: cls._validate_positive_int(x, 'num_layers')),
            'num_layers': ('model', 'num_layers', int, lambda x: cls._validate_positive_int(x, 'num_layers')),
            'dropout': ('model', 'dropout', float, lambda x: cls._validate_non_negative_float(x, 'dropout')),
            'max-seq-length': ('model', 'max_seq_length', int, lambda x: cls._validate_positive_int(x, 'max_seq_length')),
            'max_seq_length': ('model', 'max_seq_length', int, lambda x: cls._validate_positive_int(x, 'max_seq_length')),
            'output-dim': ('model', 'output_dim', int, lambda x: cls._validate_positive_int(x, 'output_dim')),
            'output_dim': ('model', 'output_dim', int, lambda x: cls._validate_positive_int(x, 'output_dim')),
            'forecast-horizon': ('model', 'forecast_horizon', int, lambda x: cls._validate_positive_int(x, 'forecast_horizon')),
            'forecast_horizon': ('model', 'forecast_horizon', int, lambda x: cls._validate_positive_int(x, 'forecast_horizon')),
            'use-attention-pooling': ('model', 'use_attention_pooling', bool, None),
            'use_attention_pooling': ('model', 'use_attention_pooling', bool, None),
            
            # Optimizer parameters
            'learning_rate': ('optimizer', 'learning_rate', float, lambda x: cls._validate_positive_float(x, 'learning_rate')),
            'learning-rate': ('optimizer', 'learning_rate', float, lambda x: cls._validate_positive_float(x, 'learning_rate')),
            'lr': ('optimizer', 'learning_rate', float, lambda x: cls._validate_positive_float(x, 'learning_rate')),
            'weight_decay': ('optimizer', 'weight_decay', float, lambda x: cls._validate_non_negative_float(x, 'weight_decay')),
            'weight-decay': ('optimizer', 'weight_decay', float, lambda x: cls._validate_non_negative_float(x, 'weight_decay')),
            'optimizer_name': ('optimizer', 'name', str, None),
            'optimizer-name': ('optimizer', 'name', str, None),
            'amsgrad': ('optimizer', 'amsgrad', bool, None),
            
            # Scheduler parameters
            'scheduler_name': ('scheduler', 'name', str, None),
            'scheduler-name': ('scheduler', 'name', str, None),
            'warmup_steps': ('scheduler', 'warmup_steps', int, lambda x: cls._validate_non_negative_int(x, 'warmup_steps')),
            'warmup-steps': ('scheduler', 'warmup_steps', int, lambda x: cls._validate_non_negative_int(x, 'warmup_steps')),
            'max_steps': ('scheduler', 'max_steps', int, lambda x: cls._validate_positive_int(x, 'max_steps')),
            'max-steps': ('scheduler', 'max_steps', int, lambda x: cls._validate_positive_int(x, 'max_steps')),
            'min_lr': ('scheduler', 'min_lr', float, lambda x: cls._validate_non_negative_float(x, 'min_lr')),
            'min-lr': ('scheduler', 'min_lr', float, lambda x: cls._validate_non_negative_float(x, 'min_lr')),
            'patience': ('scheduler', 'patience', int, lambda x: cls._validate_positive_int(x, 'patience')),
            'factor': ('scheduler', 'factor', float, lambda x: cls._validate_positive_float(x, 'factor')),
        }
        
        # Process each argument
        for arg_name, arg_value in args_dict.items():
            if arg_value is None:
                continue  # Skip None values (not provided)
            
            if arg_name in param_mapping:
                section, param_name, type_converter, validator = param_mapping[arg_name]
                
                try:
                    # Convert type
                    if type_converter == bool:
                        # Handle boolean conversion from string
                        if isinstance(arg_value, str):
                            converted_value = arg_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            converted_value = bool(arg_value)
                    elif type_converter == list:
                        # Handle list conversion
                        if isinstance(arg_value, str):
                            # Split comma-separated string
                            converted_value = [s.strip() for s in arg_value.split(',')]
                        elif isinstance(arg_value, list):
                            converted_value = arg_value
                        else:
                            converted_value = [arg_value]
                    else:
                        converted_value = type_converter(arg_value)
                    
                    # Validate if validator provided
                    if validator:
                        converted_value = validator(converted_value)
                    
                    # Store in appropriate section
                    if section == 'config':
                        config_params[param_name] = converted_value
                    elif section == 'model':
                        model_params[param_name] = converted_value
                    elif section == 'optimizer':
                        optimizer_params[param_name] = converted_value
                    elif section == 'scheduler':
                        scheduler_params[param_name] = converted_value
                    elif section == 'loss':
                        loss_params[param_name] = converted_value
                        
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for parameter '{arg_name}': {arg_value}. Error: {e}")
        
        # Create nested configuration objects
        model_config = ModelConfig(**model_params)
        optimizer_config = OptimizerConfig(**optimizer_params)
        scheduler_config = SchedulerConfig(**scheduler_params)
        loss_config = LossConfig(**loss_params)
        
        # Create and return main configuration
        try:
            return cls(
                model=model_config,
                optimizer=optimizer_config,
                scheduler=scheduler_config,
                loss=loss_config,
                **config_params
            )
        except Exception as e:
            raise ValueError(f"Failed to create TrainingConfig: {e}")
    
    @staticmethod
    def _validate_positive_int(value: int, param_name: str) -> int:
        """Validate that an integer parameter is positive."""
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")
        return value
    
    @staticmethod
    def _validate_non_negative_int(value: int, param_name: str) -> int:
        """Validate that an integer parameter is non-negative."""
        if value < 0:
            raise ValueError(f"{param_name} must be non-negative, got {value}")
        return value
    
    @staticmethod
    def _validate_positive_float(value: float, param_name: str) -> float:
        """Validate that a float parameter is positive."""
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")
        return value
    
    @staticmethod
    def _validate_non_negative_float(value: float, param_name: str) -> float:
        """Validate that a float parameter is non-negative."""
        if value < 0:
            raise ValueError(f"{param_name} must be non-negative, got {value}")
        return value
    
    @staticmethod
    def _validate_split(value: float, param_name: str) -> float:
        """Validate that a split parameter is between 0 and 1."""
        if not (0 < value < 1):
            raise ValueError(f"{param_name} must be between 0 and 1, got {value}")
        return value
    
    @staticmethod
    def _validate_device(value: str) -> str:
        """Validate that device is supported."""
        valid_devices = ["cpu", "cuda", "mps"]
        if value not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got '{value}'")
        return value
    
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


def get_parameter_mapping_documentation() -> str:
    """
    Get comprehensive documentation of parameter mappings for from_args method.
    
    Returns:
        String containing detailed parameter mapping documentation
    """
    return """
TRAININGCONFIG PARAMETER MAPPING DOCUMENTATION
==============================================

The TrainingConfig.from_args() method accepts command-line arguments or dictionary parameters
and maps them to the appropriate configuration fields. This enables easy configuration from
command-line scripts while maintaining backward compatibility.

BASIC TRAINING PARAMETERS:
--------------------------
--epochs, --num-epochs          -> num_epochs (int, >0)
--batch-size, --batch_size      -> batch_size (int, >0)
--gradient-accumulation-steps   -> gradient_accumulation_steps (int, >0)
--gradient-clip                 -> gradient_clip (float, >0)

DEVICE AND PERFORMANCE:
-----------------------
--device                        -> device (str: 'cpu', 'cuda', 'mps')
--use-amp, --use_amp            -> use_amp (bool)
--num-workers, --num_workers    -> num_workers (int, >=0)
--pin-memory, --pin_memory      -> pin_memory (bool)

CHECKPOINTING:
--------------
--save-every, --save_every      -> save_every (int, >0)
--checkpoint-dir                -> checkpoint_dir (str)
--save-best-only                -> save_best_only (bool)

EARLY STOPPING:
---------------
--early-stopping-patience       -> early_stopping_patience (int, >0)
--early-stopping-min-delta      -> early_stopping_min_delta (float, >=0)

EXPERIMENT TRACKING:
--------------------
--experiment-name               -> experiment_name (str)
--project-name                  -> project_name (str)
--log-every                     -> log_every (int, >0)

VALIDATION:
-----------
--val-every                     -> val_every (int, >0)
--val-metric                    -> val_metric (str)

DATA SPLITS:
------------
--train-split                   -> train_split (float, 0<x<1)
--val-split                     -> val_split (float, 0<x<1)
--test-split                    -> test_split (float, 0<x<1)

REPRODUCIBILITY:
----------------
--seed                          -> seed (int, >=0)
--deterministic                 -> deterministic (bool)

OPTIMIZER PARAMETERS:
---------------------
--learning-rate, --lr           -> optimizer.learning_rate (float, >0)
--weight-decay                  -> optimizer.weight_decay (float, >=0)
--optimizer-name                -> optimizer.name (str)
--amsgrad                       -> optimizer.amsgrad (bool)

SCHEDULER PARAMETERS:
---------------------
--scheduler-name                -> scheduler.name (str)
--warmup-steps                  -> scheduler.warmup_steps (int, >=0)
--max-steps                     -> scheduler.max_steps (int, >0)
--min-lr                        -> scheduler.min_lr (float, >=0)
--patience                      -> scheduler.patience (int, >0)
--factor                        -> scheduler.factor (float, >0)

BOOLEAN VALUE CONVERSION:
-------------------------
String values are converted as follows:
- 'true', '1', 'yes', 'on' -> True
- 'false', '0', 'no', 'off' -> False
- Case insensitive

USAGE EXAMPLES:
---------------
# From dictionary
config = TrainingConfig.from_args({
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'device': 'cuda'
})

# From argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()
config = TrainingConfig.from_args(args)

ERROR HANDLING:
---------------
- ValueError: Invalid parameter values (negative numbers, invalid device, etc.)
- TypeError: Wrong input type (not dict or Namespace)
- Clear error messages indicate which parameter failed and why

BACKWARD COMPATIBILITY:
-----------------------
- All existing TrainingConfig functionality remains unchanged
- Existing tests continue to pass
- Manual initialization still works: TrainingConfig(num_epochs=100, ...)
- from_yaml() and from_dict() methods still work as before
"""