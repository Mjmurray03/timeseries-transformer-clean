"""
Model loader utilities for handling different checkpoint formats.

This module provides robust functions for loading PyTorch model checkpoints
that can handle various checkpoint formats commonly used in production ML systems.
"""

import torch
from typing import Dict, Optional, Any, Union


def safe_load_checkpoint(model, model_path, device='cpu'):
    """
    Safely load a model checkpoint handling different formats.
    
    This function handles both direct state_dict saves and dictionary formats
    with 'model_state_dict' or 'state_dict' keys, ensuring compatibility with
    various checkpoint saving conventions.
    
    Args:
        model: The PyTorch model to load weights into
        model_path: Path to the checkpoint file
        device: Device to map the checkpoint to (default: 'cpu')
        
    Returns:
        The extracted state_dict from the checkpoint
        
    Example:
        >>> model = MyModel()
        >>> state_dict = safe_load_checkpoint(model, 'checkpoint.pt', device='cuda')
        >>> model.load_state_dict(state_dict)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    return state_dict


def get_model_config_from_checkpoint(model_path, device='cpu'):
    """
    Extract model configuration from a checkpoint if available.
    
    This function attempts to retrieve model configuration metadata
    from a checkpoint file, which is useful for recreating the exact
    model architecture during inference or deployment.
    
    Args:
        model_path: Path to the checkpoint file
        device: Device to map the checkpoint to (default: 'cpu')
        
    Returns:
        Model configuration dictionary if available, None otherwise
        
    Example:
        >>> config = get_model_config_from_checkpoint('checkpoint.pt')
        >>> if config:
        >>>     model = MyModel(**config)
        >>> else:
        >>>     model = MyModel()  # Use default configuration
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        return checkpoint['model_config']
        
    return None