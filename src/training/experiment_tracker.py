"""Experiment tracking integration for training pipeline."""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking across multiple platforms."""
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "timeseries-transformer",
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "runs",
        use_wandb: bool = True,
        use_mlflow: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            project_name: Name of the project
            config: Configuration dictionary to log
            log_dir: Directory for local logs
            use_wandb: Whether to use Weights & Biases
            use_mlflow: Whether to use MLflow
            use_tensorboard: Whether to use TensorBoard
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.config = config or {}
        self.log_dir = Path(log_dir)
        
        # Initialize tracking platforms
        self.wandb_run = None
        self.mlflow_run = None
        self.tb_writer = None
        
        # Setup tracking platforms
        if use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
        elif use_wandb:
            logger.warning("Weights & Biases not available, skipping W&B setup")
            
        if use_mlflow and MLFLOW_AVAILABLE:
            self._setup_mlflow()
        elif use_mlflow:
            logger.warning("MLflow not available, skipping MLflow setup")
            
        if use_tensorboard:
            self._setup_tensorboard()
        
        # Create local log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log initial config
        if self.config:
            self.log_config(self.config)
    
    def _setup_wandb(self):
        """Setup Weights & Biases tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                reinit=True
            )
            logger.info(f"Initialized W&B run: {self.wandb_run.id}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.wandb_run = None
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set experiment
            mlflow.set_experiment(self.project_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(run_name=self.experiment_name)
            logger.info(f"Initialized MLflow run: {self.mlflow_run.info.run_id}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.mlflow_run = None
    
    def _setup_tensorboard(self):
        """Setup TensorBoard tracking."""
        try:
            tb_log_dir = self.log_dir / self.experiment_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            
            self.tb_writer = SummaryWriter(str(tb_log_dir))
            logger.info(f"Initialized TensorBoard writer: {tb_log_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard: {e}")
            self.tb_writer = None
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        # W&B
        if self.wandb_run:
            try:
                wandb.config.update(config)
            except Exception as e:
                logger.error(f"Failed to log config to W&B: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                for key, value in self._flatten_dict(config).items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
            except Exception as e:
                logger.error(f"Failed to log config to MLflow: {e}")
        
        # Local JSON
        try:
            config_path = self.log_dir / f"{self.experiment_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save config locally: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to all tracking platforms.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (epoch, batch, etc.)
            prefix: Prefix for metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # W&B
        if self.wandb_run:
            try:
                log_dict = dict(metrics)
                if step is not None:
                    log_dict["step"] = step
                wandb.log(log_dict)
            except Exception as e:
                logger.error(f"Failed to log metrics to W&B: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {e}")
        
        # TensorBoard
        if self.tb_writer and step is not None:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.tb_writer.add_scalar(key, value, step)
                self.tb_writer.flush()
            except Exception as e:
                logger.error(f"Failed to log metrics to TensorBoard: {e}")
    
    def log_model(
        self,
        model_path: str,
        model_name: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log model artifacts.
        
        Args:
            model_path: Path to model file
            model_name: Name for the model
            metadata: Additional metadata
        """
        # W&B
        if self.wandb_run:
            try:
                wandb.save(model_path)
                if metadata:
                    wandb.log(metadata)
            except Exception as e:
                logger.error(f"Failed to log model to W&B: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.log_artifact(model_path)
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"model_{key}", value)
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with final metrics."""
        # TensorBoard
        if self.tb_writer:
            try:
                self.tb_writer.add_hparams(hparams, metrics)
            except Exception as e:
                logger.error(f"Failed to log hyperparameters to TensorBoard: {e}")
    
    def log_figure(self, figure, name: str, step: Optional[int] = None):
        """Log matplotlib figure."""
        # W&B
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Image(figure)}, step=step)
            except Exception as e:
                logger.error(f"Failed to log figure to W&B: {e}")
        
        # TensorBoard
        if self.tb_writer and step is not None:
            try:
                self.tb_writer.add_figure(name, figure, step)
            except Exception as e:
                logger.error(f"Failed to log figure to TensorBoard: {e}")
    
    def log_text(self, text: str, name: str, step: Optional[int] = None):
        """Log text data."""
        # W&B
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Html(text)}, step=step)
            except Exception as e:
                logger.error(f"Failed to log text to W&B: {e}")
        
        # TensorBoard
        if self.tb_writer and step is not None:
            try:
                self.tb_writer.add_text(name, text, step)
            except Exception as e:
                logger.error(f"Failed to log text to TensorBoard: {e}")
    
    def finish(self):
        """Finish experiment tracking."""
        # W&B
        if self.wandb_run:
            try:
                wandb.finish()
                logger.info("Finished W&B run")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
        
        # MLflow
        if self.mlflow_run:
            try:
                mlflow.end_run()
                logger.info("Finished MLflow run")
            except Exception as e:
                logger.error(f"Failed to finish MLflow run: {e}")
        
        # TensorBoard
        if self.tb_writer:
            try:
                self.tb_writer.close()
                logger.info("Closed TensorBoard writer")
            except Exception as e:
                logger.error(f"Failed to close TensorBoard writer: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary for logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class MetricsLogger:
    """Simple metrics logger for local tracking."""
    
    def __init__(self, log_file: str):
        """Initialize metrics logger."""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file with headers
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("timestamp,step,metric,value\n")
    
    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics to CSV file."""
        timestamp = datetime.now().isoformat()
        
        with open(self.log_file, 'a') as f:
            for metric, value in metrics.items():
                f.write(f"{timestamp},{step},{metric},{value}\n")