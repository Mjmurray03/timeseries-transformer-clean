"""Model checkpointing callback for training."""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Callback to save model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        filename: str = "checkpoint_epoch_{epoch:03d}.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        save_top_k: int = 1,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            filename: Filename template for checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitored metric
            save_best_only: Only save when metric improves
            save_last: Always save the last checkpoint
            save_top_k: Number of best checkpoints to keep
            verbose: Whether to print save messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.verbose = verbose
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self.best_score = None
        self.saved_checkpoints = []
        
        # Set comparison function
        if mode == "min":
            self.monitor_op = lambda current, best: current < best
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        logger.info(f"Initialized ModelCheckpoint: {checkpoint_dir}")
        logger.info(f"Monitoring: {monitor} ({mode})")
    
    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state to save
            metrics: Dictionary of metrics
            scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler for mixed precision (optional)
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            if self.verbose:
                logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        # Check if this is the best model
        is_best = False
        if self.best_score is None:
            is_best = True
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            is_best = True
            self.best_score = current_score
        
        # Save checkpoint if conditions are met
        should_save = (not self.save_best_only) or is_best
        
        if should_save:
            checkpoint_path = self.checkpoint_dir / self.filename.format(epoch=epoch)
            self._save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                metrics,
                scheduler,
                scaler
            )
            
            # Track saved checkpoints
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'score': current_score,
                'is_best': is_best
            })
            
            # Clean up old checkpoints if needed
            self._cleanup_checkpoints()
        
        # Always save the last checkpoint if requested
        if self.save_last:
            last_path = self.checkpoint_dir / "last_checkpoint.pt"
            self._save_checkpoint(
                last_path,
                epoch,
                model,
                optimizer,
                metrics,
                scheduler,
                scaler
            )
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            self._save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                metrics,
                scheduler,
                scaler
            )
            
            if self.verbose:
                logger.info(f"New best model saved: {self.monitor}={current_score:.6f}")
    
    def _save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        """Save checkpoint to file."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'monitor': self.monitor,
            'best_score': self.best_score
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        try:
            torch.save(checkpoint, filepath)
            if self.verbose:
                logger.info(f"Saved checkpoint: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to keep only top-k."""
        if self.save_top_k <= 0:
            return
        
        # Sort checkpoints by score
        if self.mode == "min":
            self.saved_checkpoints.sort(key=lambda x: x['score'])
        else:
            self.saved_checkpoints.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove excess checkpoints
        while len(self.saved_checkpoints) > self.save_top_k:
            checkpoint_to_remove = self.saved_checkpoints.pop()
            
            # Don't remove if it's the best checkpoint
            if not checkpoint_to_remove['is_best']:
                try:
                    os.remove(checkpoint_to_remove['path'])
                    if self.verbose:
                        logger.info(f"Removed old checkpoint: {checkpoint_to_remove['path']}")
                except OSError as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")
    
    def load_best_checkpoint(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load weights into
            
        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Best {self.monitor}: {checkpoint['best_score']:.6f}")
        
        return checkpoint
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved."""
        return self.best_score
    
    def list_checkpoints(self) -> list:
        """List all saved checkpoints."""
        return self.saved_checkpoints.copy()