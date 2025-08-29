"""
Utility functions for the time-series transformer project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_info(model: torch.nn.Module, metrics: Dict, path: str) -> None:
    """Save model information and training metrics."""
    model_info = {
        'architecture': model.__class__.__name__,
        'parameters': count_parameters(model),
        'config': getattr(model, 'get_model_info', lambda: {})(),
        'metrics': metrics,
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(model_info, f, indent=2)


def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (RMSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add minimum validation loss annotation
    min_val_epoch = np.argmin(val_losses) + 1
    min_val_loss = min(val_losses)
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_epoch}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch + len(epochs) * 0.1, min_val_loss + (max(val_losses) - min(val_losses)) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Actual vs Predicted",
    save_path: Optional[str] = None
) -> None:
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 8))
    
    # Time series plot
    plt.subplot(2, 1, 1)
    x = range(len(actual))
    plt.plot(x, actual, 'b-', label='Actual', alpha=0.8, linewidth=1.5)
    plt.plot(x, predicted, 'r-', label='Predicted', alpha=0.8, linewidth=1.5)
    plt.title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(actual, predicted, alpha=0.6, color='blue', s=20)
    
    # Perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'{title} - Scatter Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
    
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {save_path}")
    
    plt.show()


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = np.corrcoef(actual, predicted)[0, 1] ** 2 if len(np.unique(actual)) > 1 else 0.0
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2)
    }


def save_metrics(metrics: Dict, path: str) -> None:
    """Save metrics to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {path}")


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str = 'cpu') -> Dict:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        return {'model_state_dict': checkpoint}


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: str
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save model state."""
        self.best_weights = model.state_dict().copy()


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> None:
    """Print model architecture summary."""
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Input shape: {input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model configuration
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print("\nModel Configuration:")
        for key, value in info.items():
            if key != 'total_parameters':  # Already printed above
                print(f"  {key}: {value}")
    
    print("=" * 80)