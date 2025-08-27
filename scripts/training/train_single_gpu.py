#!/usr/bin/env python3
"""
GPU Training Script for Time-Series Transformer

This script implements the main training loop with:
- GPU optimization and mixed precision training
- Distributed Data Parallel (DDP) support
- Comprehensive checkpoint management
- W&B experiment tracking with custom metrics
- Learning rate scheduling with warmup
- Comprehensive error handling and recovery

Follows exact patterns from .kiro/steering/ml-infrastructure.md
"""

import os
import sys
import logging
import time
import argparse
import signal
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import math
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.training_config import TrainingConfig
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.models.losses.composite_loss import CompositeLoss
from src.data.datasets.stock_dataset import StockSequenceDataset
from src.training.experiment_tracker import ExperimentTracker
from src.training.callbacks.early_stopping import EarlyStopping
from src.training.callbacks.model_checkpoint import ModelCheckpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class LinearWarmupScheduler:
    """Linear warmup scheduler followed by cosine annealing."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class GPUTrainer:
    """Main GPU training orchestrator following ml-infrastructure.md patterns."""
    
    def __init__(self, config: TrainingConfig, rank: int = 0, world_size: int = 1):
        """
        Initialize GPU trainer.
        
        Args:
            config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Set reproducibility
        self._set_deterministic_training()
        
        # Initialize model
        self.model = self._build_model()
        
        # Setup optimizer and schedulers
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.warmup_scheduler = self._build_warmup_scheduler()
        
        # Setup loss function
        self.criterion = self._build_criterion()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Setup experiment tracking (only on rank 0)
        if self.rank == 0:
            self._setup_experiment_tracking()
            self._setup_callbacks()
        
        # GPU memory optimization
        self._optimize_gpu_memory()
        
        logger.info(f"Initialized GPUTrainer on device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
        logger.info(f"Distributed training: {self.is_distributed} (rank {rank}/{world_size})")
    
    def _set_deterministic_training(self):
        """Set deterministic training for reproducibility."""
        if self.config.deterministic:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            
            # Make CuDNN deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Enable CuDNN benchmark for better performance
            torch.backends.cudnn.benchmark = True
    
    def _build_model(self) -> nn.Module:
        """Build and initialize model."""
        model = TimeSeriesTransformer(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout,
            max_seq_length=self.config.model.max_seq_length,
            output_dim=self.config.model.output_dim,
            forecast_horizon=self.config.model.forecast_horizon,
            quantiles=self.config.model.quantiles,
            use_attention_pooling=self.config.model.use_attention_pooling
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False
            )
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer following ml-infrastructure.md patterns."""
        optimizer_config = self.config.optimizer
        
        # Get model parameters (handle DDP)
        model = self.model.module if self.is_distributed else self.model
        
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            amsgrad=optimizer_config.amsgrad
        )
        
        logger.info(f"Built optimizer: AdamW with lr={optimizer_config.learning_rate}")
        return optimizer
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler."""
        scheduler_config = self.config.scheduler
        
        if scheduler_config.name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.max_steps,
                eta_min=scheduler_config.min_lr
            )
        elif scheduler_config.name.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.patience,
                factor=scheduler_config.factor,
                min_lr=scheduler_config.min_lr,
                verbose=True
            )
        else:
            scheduler = None
        
        logger.info(f"Built scheduler: {scheduler_config.name}")
        return scheduler
    
    def _build_warmup_scheduler(self) -> Optional[LinearWarmupScheduler]:
        """Build warmup scheduler."""
        warmup = getattr(self.config, "warmup_steps", 0)
        if warmup > 0:
            total_steps = self.config.num_epochs * self.config.steps_per_epoch
            return LinearWarmupScheduler(
                self.optimizer,
                warmup_steps=warmup,
                total_steps=total_steps,
                min_lr=self.config.scheduler.min_lr
            )
        return None
    
    def _build_criterion(self) -> nn.Module:
        """Build loss function."""
        if self.config.loss.use_composite_loss:
            criterion = CompositeLoss(
                price_weight=self.config.loss.price_weight,
                direction_weight=self.config.loss.direction_weight,
                volatility_weight=self.config.loss.volatility_weight,
                quantile_weight=self.config.loss.quantile_weight,
                quantile_levels=self.config.model.quantiles
            )
        else:
            criterion = nn.MSELoss()
        
        return criterion.to(self.device)
    
    def _setup_experiment_tracking(self):
        """Setup W&B experiment tracking (only on rank 0)."""
        # Initialize wandb
        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            config=self.config.to_dict(),
            tags=["gpu_training", f"v{self.config.model.model_version}"],
            resume="allow"
        )
        
        # Setup experiment tracker
        self.tracker = ExperimentTracker(
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
            config=self.config.to_dict()
        )
        
        # Log model architecture
        model = self.model.module if self.is_distributed else self.model
        wandb.watch(model, log="all", log_freq=100)
        
        logger.info("Initialized experiment tracking")
    
    def _setup_callbacks(self):
        """Setup training callbacks (only on rank 0)."""
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode='min'
        )
        
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=self.config.save_best_only,
            monitor='val_loss',
            mode='min',
            save_top_k=3
        )
        
        logger.info("Initialized training callbacks")
    
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Log GPU memory info
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                       f"Cached: {memory_cached:.2f}GB, Total: {memory_total:.2f}GB")
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        model = self.model.module if self.is_distributed else self.model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop with GPU optimization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting GPU training...")
        self.training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler
                if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                # Training phase
                train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                if epoch % self.config.val_every == 0:
                    val_metrics = self._validate_epoch(val_loader)
                else:
                    val_metrics = {}
                
                # Learning rate scheduling
                self._update_learning_rate(val_metrics)
                
                # Only handle callbacks on rank 0
                if self.rank == 0:
                    # Checkpointing
                    if val_metrics and 'loss' in val_metrics:
                        self._save_checkpoint(epoch, val_metrics)
                        
                        # Update best validation loss
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                    
                    # Early stopping
                    if val_metrics and 'loss' in val_metrics:
                        if self.early_stopping.should_stop(val_metrics['loss']):
                            logger.info(f"Early stopping at epoch {epoch}")
                            break
                    
                    # Experiment tracking
                    self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Synchronize processes
                if self.is_distributed:
                    dist.barrier()
                
                # Progress logging
                if self.rank == 0 and (epoch % 10 == 0 or epoch == self.config.num_epochs - 1):
                    self._log_progress(epoch, train_metrics, val_metrics)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.rank == 0:
                self._save_checkpoint(self.current_epoch, {}, prefix="interrupted")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            if self.rank == 0:
                self._save_checkpoint(self.current_epoch, {}, prefix="failed")
            raise
        
        finally:
            results = self._finalize_training(test_loader)
            return results
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Single epoch training with mixed precision and gradient accumulation.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_loss_components = defaultdict(list)
        
        # Progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(
                data_loader,
                desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
                leave=False
            )
        else:
            pbar = data_loader
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device
                inputs = batch['inputs'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                with autocast(enabled=self.config.use_amp):
                    # Forward pass
                    predictions = self.model(inputs)
                    
                    # Calculate loss
                    if self.config.loss.use_composite_loss:
                        # Composite loss expects dictionaries
                        if isinstance(predictions, dict):
                            loss, loss_components = self.criterion(predictions, targets)
                        else:
                            # Convert single output to dict format
                            pred_dict = {'price': predictions}
                            target_dict = {'price': targets, 'volatility': targets}  # Placeholder
                            loss, loss_components = self.criterion(pred_dict, target_dict)
                    else:
                        # Simple MSE loss
                        if isinstance(predictions, dict):
                            loss = self.criterion(predictions['price'], targets)
                        else:
                            loss = self.criterion(predictions, targets)
                        loss_components = {'total_loss': loss.item()}
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update learning rate with warmup
                    if self.warmup_scheduler:
                        self.warmup_scheduler.step()
                    
                    # Update global step
                    self.global_step += 1
                
                # Track metrics
                epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
                for k, v in loss_components.items():
                    epoch_loss_components[k].append(v)
                
                # Update progress bar (only on rank 0)
                if self.rank == 0 and batch_idx % 10 == 0:
                    current_loss = np.mean(epoch_losses[-100:])
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss': f'{current_loss:.6f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Log batch metrics (only on rank 0)
                if self.rank == 0 and self.global_step % self.config.log_every == 0:
                    batch_metrics = {
                        'batch_loss': loss.item() * self.config.gradient_accumulation_steps,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'gradient_norm': self._get_gradient_norm()
                    }
                    self.tracker.log_metrics(batch_metrics, self.global_step, prefix="batch")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM error at batch {batch_idx}, reducing batch size")
                    torch.cuda.empty_cache()
                    # Skip this batch
                    continue
                else:
                    raise e
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses),
            **{k: np.mean(v) for k, v in epoch_loss_components.items()}
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def _validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validation loop.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        val_loss_components = defaultdict(list)
        all_predictions = []
        all_targets = []
        
        # Progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(data_loader, desc="Validating", leave=False)
        else:
            pbar = data_loader
        
        for batch in pbar:
            try:
                # Move batch to device
                inputs = batch['inputs'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    predictions = self.model(inputs)
                    
                    # Calculate loss
                    if self.config.loss.use_composite_loss:
                        if isinstance(predictions, dict):
                            loss, loss_components = self.criterion(predictions, targets)
                        else:
                            pred_dict = {'price': predictions}
                            target_dict = {'price': targets, 'volatility': targets}
                            loss, loss_components = self.criterion(pred_dict, target_dict)
                    else:
                        if isinstance(predictions, dict):
                            loss = self.criterion(predictions['price'], targets)
                        else:
                            loss = self.criterion(predictions, targets)
                        loss_components = {'total_loss': loss.item()}
                
                val_losses.append(loss.item())
                for k, v in loss_components.items():
                    val_loss_components[k].append(v)
                
                # Store for additional metrics calculation
                if isinstance(predictions, dict):
                    pred_values = predictions['price']
                else:
                    pred_values = predictions
                
                all_predictions.append(pred_values.cpu())
                all_targets.append(targets.cpu())
                
                # Update progress bar (only on rank 0)
                if self.rank == 0 and len(val_losses) % 10 == 0:
                    pbar.set_postfix({'val_loss': f'{np.mean(val_losses):.6f}'})
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning("OOM error during validation, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Calculate comprehensive metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            additional_metrics = self._calculate_comprehensive_metrics(all_predictions, all_targets)
        else:
            additional_metrics = {}
        
        # Combine all metrics
        val_metrics = {
            'loss': np.mean(val_losses),
            'loss_std': np.std(val_losses),
            **{k: np.mean(v) for k, v in val_loss_components.items()},
            **additional_metrics
        }
        
        return val_metrics
    
    def _calculate_comprehensive_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics following requirements.md.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Basic regression metrics
        mse = np.mean((pred_np - target_np) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_np - target_np))
        
        # Directional accuracy (for multi-step predictions)
        if pred_np.shape[1] > 1:
            pred_direction = np.diff(pred_np, axis=1) > 0
            target_direction = np.diff(target_np, axis=1) > 0
            directional_accuracy = np.mean(pred_direction == target_direction)
        else:
            directional_accuracy = 0.0
        
        # Financial metrics (simplified)
        returns = np.diff(pred_np, axis=1) if pred_np.shape[1] > 1 else np.zeros_like(pred_np)
        if returns.size > 0:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(returns)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        if returns.size == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns.flatten())
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def _update_learning_rate(self, val_metrics: Dict[str, float]):
        """Update learning rate with scheduler."""
        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if 'loss' in val_metrics:
                    self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
    
    def _get_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], prefix: str = ""):
        """Save training checkpoint with comprehensive state."""
        checkpoint_name = f"{prefix}_epoch_{epoch}.pt" if prefix else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (handle DDP)
        model = self.model.module if self.is_distributed else self.model
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
            'rng_states': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.warmup_scheduler:
            checkpoint['warmup_scheduler_state'] = {
                'current_step': self.warmup_scheduler.current_step
            }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Keep only last 3 checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        # Get all checkpoint files
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove old checkpoints (keep last 3)
        for checkpoint in checkpoints[:-3]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint}")
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to experiment trackers."""
        # Log to wandb
        log_dict = {}
        
        # Training metrics
        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = v
        
        # Validation metrics
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
        
        # System metrics
        log_dict.update({
            "epoch": epoch,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "global_step": self.global_step
        })
        
        # GPU metrics
        if torch.cuda.is_available():
            log_dict.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**3,
                "gpu_memory_cached": torch.cuda.memory_reserved(self.device) / 1024**3
            })
        
        wandb.log(log_dict, step=epoch)
        
        # Log to experiment tracker
        self.tracker.log_metrics(train_metrics, epoch, prefix="train")
        if val_metrics:
            self.tracker.log_metrics(val_metrics, epoch, prefix="val")
    
    def _log_progress(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log training progress."""
        train_loss = train_metrics.get('loss', 0.0)
        val_loss = val_metrics.get('loss', 0.0) if val_metrics else 0.0
        lr = self.optimizer.param_groups[0]['lr']
        
        elapsed_time = time.time() - self.training_start_time
        
        logger.info(
            f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )
    
    def _finalize_training(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Finalize training and return results."""
        results = {}
        
        if self.rank == 0:
            # Final evaluation on test set
            if test_loader:
                logger.info("Running final evaluation on test set...")
                test_metrics = self._validate_epoch(test_loader)
                results['test_metrics'] = test_metrics
                self.tracker.log_metrics(test_metrics, self.current_epoch, prefix="test")
            
            # Training summary
            training_time = time.time() - self.training_start_time if self.training_start_time else 0
            results.update({
                'training_time': training_time,
                'epochs_completed': self.current_epoch + 1,
                'best_val_loss': self.best_val_loss,
                'final_lr': self.optimizer.param_groups[0]['lr'],
                'total_steps': self.global_step
            })
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
            logger.info(f"Total training steps: {self.global_step}")
            
            # Save final checkpoint
            self._save_checkpoint(self.current_epoch, results.get('test_metrics', {}), prefix="final")
            
            # Finish experiment tracking
            self.tracker.finish()
            wandb.finish()
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint for resuming training."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        model = self.model.module if self.is_distributed else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load warmup scheduler state
        if self.warmup_scheduler and 'warmup_scheduler_state' in checkpoint:
            self.warmup_scheduler.current_step = checkpoint['warmup_scheduler_state']['current_step']
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore RNG states for reproducibility
        if 'rng_states' in checkpoint:
            rng_states = checkpoint['rng_states']
            random.setstate(rng_states['python'])
            np.random.set_state(rng_states['numpy'])
            torch.set_rng_state(rng_states['torch'])
            if torch.cuda.is_available() and rng_states['torch_cuda'] is not None:
                torch.cuda.set_rng_state(rng_states['torch_cuda'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")
        return checkpoint


def setup_distributed_training() -> Tuple[int, int]:
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://'
        )
        
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
        return rank, world_size
    else:
        return 0, 1


def create_data_loaders(config: TrainingConfig, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create data loaders with real stock data and distributed sampling."""
    from pathlib import Path
    import pandas as pd
    from src.data.processors.feature_engineering import FeatureEngineer
    from src.data.datasets.stock_dataset import StockSequenceDataset
    
    # Configuration
    batch_size = config.batch_size
    window_size = config.model.max_seq_length
    forecast_horizon = config.model.forecast_horizon
    
    # Load stock data from parquet files
    data_dir = Path("data/raw")
    all_data = []
    
    # Default to AAPL if no tickers specified
    tickers = getattr(config, 'tickers', ['AAPL'])
    if isinstance(tickers, str):
        tickers = [tickers]
    
    logger.info(f"Loading data for tickers: {tickers}")
    
    for ticker in tickers:
        ticker_dir = data_dir / ticker
        if not ticker_dir.exists():
            logger.warning(f"Data directory not found for {ticker}: {ticker_dir}")
            continue
            
        # Find the most recent parquet file for this ticker
        parquet_files = list(ticker_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found for {ticker}")
            continue
            
        # Use the most recent file (by modification time)
        latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading {ticker} data from: {latest_file}")
        
        try:
            ticker_data = pd.read_parquet(latest_file)
            ticker_data['Ticker'] = ticker  # Ensure ticker column exists
            all_data.append(ticker_data)
        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {e}")
            continue
    
    if not all_data:
        raise FileNotFoundError("No valid stock data files found. Please ensure data exists in data/raw/<TICKER>/*.parquet")
    
    # Combine all ticker data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_data.shape}")
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer()
    feature_data = feature_engineer.engineer_features(combined_data)
    logger.info(f"Feature-engineered data shape: {feature_data.shape}")
    
    # Create sequences from the feature data
    sequences, targets = create_sequences(
        feature_data, 
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )
    logger.info(f"Created {len(sequences)} sequences")
    
    # Split data into train/val/test
    train_sequences, val_sequences, test_sequences, train_targets, val_targets, test_targets = split_sequences(
        sequences, targets, config.train_split, config.val_split, config.test_split
    )
    
    # Create datasets
    train_dataset = StockSequenceDataset(train_sequences, train_targets)
    val_dataset = StockSequenceDataset(val_sequences, val_targets) 
    test_dataset = StockSequenceDataset(test_sequences, test_targets)
    
    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # Create data loaders
    def collate_fn(batch):
        # batch is a list of dictionaries with 'inputs' and 'targets' keys
        inputs = torch.stack([item['inputs'] for item in batch])
        targets = torch.stack([item['targets'] for item in batch])
        return {
            'inputs': inputs,
            'targets': targets
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        persistent_workers=False,  # Must be False when num_workers=0
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        persistent_workers=False,  # Must be False when num_workers=0
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True,
        persistent_workers=False,  # Must be False when num_workers=0
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def create_sequences(data: pd.DataFrame, window_size: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time-series data.
    
    Args:
        data: Feature-engineered DataFrame
        window_size: Size of input window
        forecast_horizon: Number of steps to forecast
        
    Returns:
        Tuple of (sequences, targets) arrays
    """
    # Remove non-numeric columns and handle missing values
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Drop rows with any NaN values
    clean_data = numeric_data.dropna()
    
    if len(clean_data) < window_size + forecast_horizon:
        raise ValueError(f"Not enough data points: {len(clean_data)} < {window_size + forecast_horizon}")
    
    sequences = []
    targets = []
    
    for i in range(len(clean_data) - window_size - forecast_horizon + 1):
        # Input sequence
        seq = clean_data.iloc[i:i + window_size].values
        sequences.append(seq)
        
        # Target sequence (Close price for forecast horizon)
        if 'Close' in clean_data.columns:
            target = clean_data['Close'].iloc[i + window_size:i + window_size + forecast_horizon].values
        else:
            # Fallback to first column if Close not found
            target = clean_data.iloc[i + window_size:i + window_size + forecast_horizon, 0].values
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def split_sequences(
    sequences: np.ndarray, 
    targets: np.ndarray, 
    train_split: float, 
    val_split: float, 
    test_split: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train/val/test sets.
    
    Args:
        sequences: Input sequences
        targets: Target sequences
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        
    Returns:
        Tuple of (train_seq, val_seq, test_seq, train_targets, val_targets, test_targets)
    """
    n_samples = len(sequences)
    
    # Calculate split indices
    train_idx = int(n_samples * train_split)
    val_idx = int(n_samples * (train_split + val_split))
    
    # Split sequences
    train_sequences = sequences[:train_idx]
    val_sequences = sequences[train_idx:val_idx]
    test_sequences = sequences[val_idx:]
    
    # Split targets
    train_targets = targets[:train_idx]
    val_targets = targets[train_idx:val_idx] 
    test_targets = targets[val_idx:]
    
    return train_sequences, val_sequences, test_sequences, train_targets, val_targets, test_targets


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments following deployment-standards.md."""
    parser = argparse.ArgumentParser(
        description="GPU Training Script for Time-Series Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--config', type=str, default='configs/training/default.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model/transformer_base.yaml',
                       help='Path to model configuration file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Model hidden dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    
    # Training optimization
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Number of warmup steps')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-every', type=int, default=100,
                       help='Log metrics every N steps')
    parser.add_argument('--val-every', type=int, default=1,
                       help='Validate every N epochs')
    
    # Experiment tracking
    parser.add_argument('--project-name', type=str, default='timeseries-transformer',
                       help='W&B project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--tags', nargs='+', default=[],
                       help='Experiment tags')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='Stock tickers to train on')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic training')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device')
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle training interruption gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Cleanup will be handled in the finally block of the training loop


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed training
    rank, world_size = setup_distributed_training()
    
    try:
        # Create training configuration
        config = TrainingConfig.from_args(args)
        
        # Generate experiment name if not provided
        if not config.experiment_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config.experiment_name = f"transformer_gpu_{timestamp}"
        
        # Create trainer
        trainer = GPUTrainer(config, rank=rank, world_size=world_size)
        
        # Load checkpoint if resuming
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config, rank, world_size)
        
        # Start training
        results = trainer.train(train_loader, val_loader, test_loader)
        
        # Log final results (only on rank 0)
        if rank == 0:
            logger.info("Training Results:")
            for key, value in results.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for k, v in value.items():
                        logger.info(f"  {k}: {v}")
                else:
                    logger.info(f"{key}: {value}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup distributed training
        if world_size > 1:
            dist.destroy_process_group()
        
        logger.info("Training completed successfully")


if __name__ == "__main__":
    main()