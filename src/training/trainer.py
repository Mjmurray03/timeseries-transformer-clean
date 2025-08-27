"""Training orchestrator for time-series transformer."""

import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from ..config.training_config import TrainingConfig
from .experiment_tracker import ExperimentTracker
from .callbacks.early_stopping import EarlyStopping
from .callbacks.model_checkpoint import ModelCheckpoint

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Manages complete training pipeline with mixed precision and gradient accumulation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize training orchestrator.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to use for training
        """
        self.config = config
        self.device = device or torch.device(config.device)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Setup training components
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None
        
        # Experiment tracking
        self.tracker = ExperimentTracker(
            experiment_name=config.experiment_name,
            project_name=config.project_name,
            config=config.to_dict()
        )
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'
        )
        
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=config.save_best_only,
            monitor='val_loss',
            mode='min'
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Initialized TrainingOrchestrator with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        optimizer_config = self.config.optimizer
        
        if optimizer_config.name.lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                amsgrad=optimizer_config.amsgrad
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
        
        logger.info(f"Built optimizer: {optimizer_config.name}")
        return optimizer
    
    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler from config."""
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
                min_lr=scheduler_config.min_lr
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")
        
        logger.info(f"Built scheduler: {scheduler_config.name}")
        return scheduler
    
    def _build_criterion(self) -> nn.Module:
        """Build loss function from config."""
        # For now, use simple MSE loss
        # This will be replaced with CompositeLoss in TASK-T006
        criterion = nn.MSELoss()
        logger.info("Built criterion: MSELoss")
        return criterion
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        # Set reproducibility
        if self.config.deterministic:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch(train_loader)
                
                # Validation phase
                if epoch % self.config.val_every == 0:
                    val_metrics = self.validate(val_loader)
                else:
                    val_metrics = {}
                
                # Learning rate scheduling
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if 'loss' in val_metrics:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                # Checkpointing
                if val_metrics and 'loss' in val_metrics:
                    self.checkpoint_callback.on_epoch_end(
                        epoch, self.model, self.optimizer, val_metrics
                    )
                    
                    # Update best validation loss
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                
                # Early stopping
                if val_metrics and 'loss' in val_metrics:
                    if self.early_stopping.should_stop(val_metrics['loss']):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Logging
                self.tracker.log_metrics(train_metrics, epoch, prefix="train")
                if val_metrics:
                    self.tracker.log_metrics(val_metrics, epoch, prefix="val")
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tracker.log_metrics({'learning_rate': current_lr}, epoch)
                
                # Progress logging
                if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                    self._log_progress(epoch, train_metrics, val_metrics)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final evaluation
            results = {}
            if test_loader:
                logger.info("Running final evaluation on test set...")
                test_metrics = self.evaluate(test_loader)
                results['test_metrics'] = test_metrics
                self.tracker.log_metrics(test_metrics, self.current_epoch, prefix="test")
            
            # Training summary
            training_time = time.time() - start_time
            results.update({
                'training_time': training_time,
                'epochs_completed': self.current_epoch + 1,
                'best_val_loss': self.best_val_loss,
                'final_lr': self.optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
            
            # Finish experiment tracking
            self.tracker.finish()
            
            return results
    
    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Single epoch training with mixed precision and gradient accumulation.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        # Progress bar
        pbar = tqdm(
            data_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=False
        )
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            inputs = batch['inputs'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast(enabled=self.config.use_amp):
                # Forward pass
                predictions = self.model(inputs)
                
                # Calculate loss
                if isinstance(predictions, dict):
                    # Multi-output model
                    loss = self.criterion(predictions['price'], targets)
                else:
                    # Single output model
                    loss = self.criterion(predictions, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
            
            # Track metrics
            epoch_losses.append(loss.item() * self.config.gradient_accumulation_steps)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_loss = np.mean(epoch_losses[-100:])
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{current_loss:.6f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Log batch metrics
            if self.global_step % self.config.log_every == 0:
                batch_metrics = {
                    'batch_loss': loss.item() * self.config.gradient_accumulation_steps,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                self.tracker.log_metrics(batch_metrics, self.global_step, prefix="batch")
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses)
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validation loop.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        
        pbar = tqdm(data_loader, desc="Validating", leave=False)
        
        for batch in pbar:
            # Move batch to device
            inputs = batch['inputs'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.config.use_amp):
                predictions = self.model(inputs)
                
                # Calculate loss
                if isinstance(predictions, dict):
                    loss = self.criterion(predictions['price'], targets)
                else:
                    loss = self.criterion(predictions, targets)
            
            val_losses.append(loss.item())
            
            # Update progress bar
            if len(val_losses) % 10 == 0:
                pbar.set_postfix({'val_loss': f'{np.mean(val_losses):.6f}'})
        
        # Calculate validation metrics
        val_metrics = {
            'loss': np.mean(val_losses),
            'loss_std': np.std(val_losses)
        }
        
        return val_metrics
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluation loop for test set.
        
        Args:
            data_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.model.eval()
        test_losses = []
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        
        for batch in pbar:
            # Move batch to device
            inputs = batch['inputs'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.config.use_amp):
                predictions = self.model(inputs)
                
                # Calculate loss
                if isinstance(predictions, dict):
                    loss = self.criterion(predictions['price'], targets)
                    pred_values = predictions['price']
                else:
                    loss = self.criterion(predictions, targets)
                    pred_values = predictions
            
            test_losses.append(loss.item())
            
            # Store predictions and targets for additional metrics
            all_predictions.append(pred_values.cpu())
            all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate comprehensive metrics
        test_metrics = self._calculate_metrics(all_predictions, all_targets)
        test_metrics['loss'] = np.mean(test_losses)
        test_metrics['loss_std'] = np.std(test_losses)
        
        return test_metrics
    
    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
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
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
    
    def _log_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log training progress."""
        train_loss = train_metrics.get('loss', 0.0)
        val_loss = val_metrics.get('loss', 0.0)
        lr = self.optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e}"
        )
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
            'global_step': self.global_step
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint