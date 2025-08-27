"""
Weights & Biases Setup Module for Time-Series Transformer Training

This module implements WANDB integration following the exact specifications
from .kiro/specs/training-pipeline/design.md and .kiro/steering/ml-infrastructure.md
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

import wandb
import torch
import numpy as np

from src.config.secrets import secrets
from src.config.config import Config

logger = logging.getLogger(__name__)


class WANDBConfig:
    """Configuration class for WANDB setup following design specifications."""
    
    # Default project configuration from ml-infrastructure.md
    DEFAULT_PROJECT = "timeseries-transformer"
    DEFAULT_ENTITY = None  # Use default WANDB entity
    
    # Experiment naming convention from ml-infrastructure.md
    EXPERIMENT_NAME_FORMAT = "{model_type}_{dataset}_{key_hyperparam}_{timestamp}"
    
    # Default tags from training pipeline specifications
    DEFAULT_TAGS = ["production", "transformer", "stock-prediction"]
    
    # Logging intervals from requirements.md
    BATCH_LOG_INTERVAL = 10  # Every 10 batches
    EPOCH_LOG_INTERVAL = 1   # Every epoch
    VALIDATION_LOG_INTERVAL = 1  # Every epoch
    CHECKPOINT_LOG_INTERVAL = 10  # Every 10 epochs
    VISUALIZATION_LOG_INTERVAL = 5   # Every 5 epochs


def init_wandb(
    project_name: str = WANDBConfig.DEFAULT_PROJECT,
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = WANDBConfig.DEFAULT_ENTITY,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    mode: Optional[str] = None,
    resume: Optional[str] = None,
    job_type: Optional[str] = "training"
) -> wandb.sdk.wandb_run.Run:
    """
    Initialize Weights & Biases tracking with Doppler secrets integration.
    
    This function implements the W&B Integration Pattern from ml-infrastructure.md
    with proper error handling and fallback mechanisms.
    
    Args:
        project_name: Name of the WANDB project (default: "timeseries-transformer")
        experiment_name: Name of the experiment run (auto-generated if None)
        config: Configuration dictionary to log to WANDB
        entity: WANDB entity/team name (uses account default if None)
        tags: List of tags for the experiment
        notes: Optional notes/description for the run
        mode: WANDB mode ("online", "offline", "disabled")
        resume: Resume behavior ("allow", "must", "never")
        job_type: Type of job ("training", "evaluation", "preprocessing")
        
    Returns:
        WANDB run object for logging metrics and artifacts
        
    Raises:
        WANDBSetupError: If WANDB initialization fails critically
        
    Example:
        >>> # Basic initialization
        >>> run = init_wandb()
        
        >>> # With custom configuration
        >>> config = {
        >>>     "architecture": "transformer",
        >>>     "dataset": "SP500",
        >>>     "epochs": 100,
        >>>     "batch_size": 32,
        >>>     "learning_rate": 1e-4
        >>> }
        >>> run = init_wandb(config=config, tags=["experiment", "v1.0.0"])
    """
    logger.info("Initializing Weights & Biases integration")
    
    # Get API key from Doppler secrets
    api_key = _get_wandb_api_key()
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = _generate_experiment_name(config)
    
    # Prepare default tags
    if tags is None:
        tags = WANDBConfig.DEFAULT_TAGS.copy()
    
    # Add timestamp and model info to tags
    tags.append(f"started_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if config and "architecture" in config:
        tags.append(f"arch_{config['architecture']}")
    
    # Prepare configuration with defaults
    wandb_config = _prepare_config(config)
    
    try:
        # Login with API key
        if api_key and mode != "offline":
            wandb.login(key=api_key, relogin=True)
            logger.info("Successfully authenticated with WANDB API key")
        elif mode != "offline":
            logger.warning("WANDB API key not found, attempting to use cached credentials")
        
        # Initialize WANDB run
        run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=wandb_config,
            entity=entity,
            tags=tags,
            notes=notes,
            mode=mode,
            resume=resume,
            job_type=job_type,
            reinit=True,
            settings=wandb.Settings(start_method="fork")  # Better process handling
        )
        
        if run is None:
            raise WANDBSetupError("WANDB initialization returned None")
        
        logger.info(f"WANDB run initialized successfully: {run.id}")
        logger.info(f"Experiment name: {experiment_name}")
        logger.info(f"Project: {project_name}")
        logger.info(f"Tags: {tags}")
        logger.info(f"WANDB dashboard: {run.url}")
        
        # Log system information
        _log_system_info(run)
        
        # Setup artifact logging
        _setup_artifact_logging(run)
        
        return run
        
    except Exception as e:
        error_msg = f"Failed to initialize WANDB: {e}"
        logger.error(error_msg)
        
        # Attempt offline fallback if online mode fails
        if mode != "offline":
            logger.warning("Attempting to initialize WANDB in offline mode")
            try:
                run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=wandb_config,
                    tags=tags,
                    notes=notes,
                    mode="offline",
                    job_type=job_type,
                    reinit=True
                )
                logger.info("WANDB initialized in offline mode")
                return run
            except Exception as fallback_error:
                logger.error(f"Offline fallback also failed: {fallback_error}")
        
        raise WANDBSetupError(error_msg) from e


def setup_wandb_for_training(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any],
    experiment_name: Optional[str] = None
) -> wandb.sdk.wandb_run.Run:
    """
    Setup WANDB specifically for training pipeline with comprehensive configuration.
    
    This follows the Training Pipeline Pattern from design.md with all required
    configuration parameters and metrics tracking setup.
    
    Args:
        model_config: Model architecture configuration
        training_config: Training hyperparameters and settings
        data_config: Data pipeline configuration
        experiment_name: Optional custom experiment name
        
    Returns:
        Initialized WANDB run configured for training
    """
    # Combine all configuration
    full_config = {
        "model": model_config,
        "training": training_config,
        "data": data_config,
        "system": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__
        }
    }
    
    # Generate training-specific tags
    tags = ["training", "transformer", "time-series"]
    
    if "dataset" in data_config:
        tags.append(f"data_{data_config['dataset']}")
    
    if "architecture" in model_config:
        tags.append(f"arch_{model_config['architecture']}")
    
    if "learning_rate" in training_config:
        lr_str = f"lr{training_config['learning_rate']:.0e}".replace("-", "n")
        tags.append(lr_str)
    
    # Add model version if available
    if hasattr(Config, 'MODEL_VERSION'):
        tags.append(f"v{Config.MODEL_VERSION}")
    
    # Initialize with training-specific configuration
    run = init_wandb(
        config=full_config,
        tags=tags,
        job_type="training",
        experiment_name=experiment_name,
        notes=f"Training run for {model_config.get('architecture', 'transformer')} model"
    )
    
    # Setup training-specific watches
    logger.info("WANDB setup complete for training pipeline")
    return run


def log_model_architecture(run: wandb.sdk.wandb_run.Run, model: torch.nn.Module) -> None:
    """
    Log model architecture details to WANDB.
    
    Args:
        run: Active WANDB run
        model: PyTorch model to analyze
    """
    try:
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        run.summary["model/total_parameters"] = total_params
        run.summary["model/trainable_parameters"] = trainable_params
        run.summary["model/non_trainable_parameters"] = total_params - trainable_params
        
        logger.info(f"Logged model architecture: {total_params:,} total parameters")
        
        # Watch model for gradient and parameter tracking
        wandb.watch(model, log="all", log_freq=WANDBConfig.BATCH_LOG_INTERVAL)
        
    except Exception as e:
        logger.error(f"Failed to log model architecture: {e}")


def create_experiment_group(
    group_name: str,
    experiments: List[str],
    description: Optional[str] = None
) -> None:
    """
    Create a group of related experiments in WANDB.
    
    Args:
        group_name: Name of the experiment group
        experiments: List of experiment names to include
        description: Optional description of the group
    """
    try:
        # This would typically be done through WANDB API or dashboard
        # For now, we log the grouping information
        logger.info(f"Creating experiment group: {group_name}")
        logger.info(f"Experiments: {experiments}")
        
        if description:
            logger.info(f"Description: {description}")
            
    except Exception as e:
        logger.error(f"Failed to create experiment group: {e}")


def _get_wandb_api_key() -> Optional[str]:
    """Get WANDB API key from Doppler secrets."""
    try:
        api_key = secrets.get('WANDB_API_KEY')
        if api_key:
            logger.debug("WANDB API key retrieved from Doppler secrets")
            return api_key
        else:
            logger.warning("WANDB_API_KEY not found in Doppler secrets")
            return None
    except Exception as e:
        logger.error(f"Error retrieving WANDB API key: {e}")
        return None


def _generate_experiment_name(config: Optional[Dict[str, Any]]) -> str:
    """
    Generate experiment name following naming convention from ml-infrastructure.md.
    
    Format: {model_type}_{dataset}_{key_hyperparam}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config is None:
        return f"transformer_default_experiment_{timestamp}"
    
    # Extract components
    model_type = "transformer"
    if "model" in config and "architecture" in config["model"]:
        model_type = config["model"]["architecture"]
    elif "architecture" in config:
        model_type = config["architecture"]
    
    dataset = "stocks"
    if "data" in config and "dataset" in config["data"]:
        dataset = config["data"]["dataset"]
    elif "dataset" in config:
        dataset = config["dataset"]
    
    key_hyperparam = ""
    if "training" in config and "learning_rate" in config["training"]:
        lr = config["training"]["learning_rate"]
        key_hyperparam = f"lr{lr:.0e}".replace("-", "n")
    elif "learning_rate" in config:
        lr = config["learning_rate"]
        key_hyperparam = f"lr{lr:.0e}".replace("-", "n")
    else:
        key_hyperparam = "default"
    
    experiment_name = f"{model_type}_{dataset}_{key_hyperparam}_{timestamp}"
    
    # Clean up name (replace invalid characters)
    experiment_name = experiment_name.replace(".", "p").replace(" ", "_").lower()
    
    return experiment_name


def _prepare_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare and validate configuration for WANDB logging."""
    if config is None:
        config = {}
    
    # Add default configuration from specifications
    default_config = {
        "framework": "pytorch",
        "project_type": "time_series_prediction",
        "model_family": "transformer",
        "domain": "financial_markets"
    }
    
    # Merge with provided config
    wandb_config = {**default_config, **config}
    
    # Ensure all values are JSON serializable
    return _sanitize_config(wandb_config)


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration values for WANDB compatibility."""
    sanitized = {}
    
    for key, value in config.items():
        try:
            if isinstance(value, dict):
                sanitized[key] = _sanitize_config(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [_sanitize_value(v) for v in value]
            else:
                sanitized[key] = _sanitize_value(value)
        except Exception as e:
            logger.warning(f"Skipping config key '{key}' due to serialization error: {e}")
    
    return sanitized


def _sanitize_value(value: Any) -> Any:
    """Sanitize individual values for JSON serialization."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif hasattr(value, '__dict__'):
        return str(value)  # Convert objects to string representation
    else:
        return str(value)


def _log_system_info(run: wandb.sdk.wandb_run.Run) -> None:
    """Log system information to WANDB run."""
    try:
        import psutil
        import platform
        
        system_info = {
            "system/platform": platform.platform(),
            "system/python_version": platform.python_version(),
            "system/cpu_count": os.cpu_count(),
            "system/memory_gb": psutil.virtual_memory().total / (1024**3),
            "system/disk_usage_gb": psutil.disk_usage('/').total / (1024**3)
        }
        
        # GPU information
        if torch.cuda.is_available():
            system_info.update({
                "system/cuda_version": torch.version.cuda,
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_name": torch.cuda.get_device_name(0)
            })
        
        run.config.update(system_info)
        logger.debug("System information logged to WANDB")
        
    except ImportError:
        logger.warning("psutil not available, skipping detailed system info")
    except Exception as e:
        logger.error(f"Failed to log system info: {e}")


def _setup_artifact_logging(run: wandb.sdk.wandb_run.Run) -> None:
    """Setup artifact logging directories and policies."""
    try:
        # Create artifacts directory if it doesn't exist
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Log artifacts directory location
        run.config.update({"artifacts_dir": str(artifacts_dir.absolute())})
        
        logger.debug("Artifact logging setup complete")
        
    except Exception as e:
        logger.error(f"Failed to setup artifact logging: {e}")


class WANDBSetupError(Exception):
    """Custom exception for WANDB setup errors."""
    pass


# Convenience functions for common use cases
def init_wandb_for_evaluation(
    model_version: str,
    test_config: Dict[str, Any],
    experiment_name: Optional[str] = None
) -> wandb.sdk.wandb_run.Run:
    """Initialize WANDB for model evaluation."""
    tags = ["evaluation", "testing", model_version]
    
    config = {
        "model_version": model_version,
        "evaluation": test_config,
        "job_type": "evaluation"
    }
    
    return init_wandb(
        config=config,
        tags=tags,
        job_type="evaluation",
        experiment_name=experiment_name or f"eval_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


def init_wandb_for_hyperparameter_search(
    search_config: Dict[str, Any],
    experiment_name: Optional[str] = None
) -> wandb.sdk.wandb_run.Run:
    """Initialize WANDB for hyperparameter optimization."""
    tags = ["hyperparameter_search", "optimization", "sweep"]
    
    config = {
        "search_space": search_config,
        "job_type": "hyperparameter_search"
    }
    
    return init_wandb(
        config=config,
        tags=tags,
        job_type="hyperparameter_search",
        experiment_name=experiment_name or f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


# Module-level convenience for backward compatibility
def setup_wandb(*args, **kwargs):
    """Deprecated: Use init_wandb instead."""
    logger.warning("setup_wandb is deprecated, use init_wandb instead")
    return init_wandb(*args, **kwargs)