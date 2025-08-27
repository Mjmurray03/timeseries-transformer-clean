"""Early stopping callback for training."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to stop training when metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = False
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        
        # Set comparison function based on mode
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
        elif mode == 'max':
            self.monitor_op = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        logger.info(f"Initialized EarlyStopping with patience={patience}, mode={mode}")
    
    def should_stop(self, current_score: float, model_weights: Optional[dict] = None) -> bool:
        """
        Check if training should stop based on current score.
        
        Args:
            current_score: Current metric value
            model_weights: Current model weights (for restoration)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            return False
        
        if self.monitor_op(current_score, self.best_score):
            # Improvement detected
            self.best_score = current_score
            self.wait = 0
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            logger.debug(f"Metric improved to {current_score:.6f}")
        else:
            # No improvement
            self.wait += 1
            logger.debug(f"No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                logger.info(f"Early stopping triggered after {self.wait} epochs without improvement")
                logger.info(f"Best score: {self.best_score:.6f}")
                return True
        
        return False
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved."""
        return self.best_score
    
    def get_best_weights(self) -> Optional[dict]:
        """Get the best model weights."""
        return self.best_weights
    
    def reset(self):
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        logger.info("Reset EarlyStopping state")