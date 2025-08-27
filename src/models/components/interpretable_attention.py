"""
Interpretable attention implementation with visualization utilities.

This module implements attention mechanisms with enhanced interpretability features,
including attention weight extraction, multi-head aggregation, and visualization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class InterpretableAttention(nn.Module):
    """
    Custom attention mechanism with interpretability features.
    
    This implementation provides detailed attention weight extraction and aggregation
    capabilities for understanding model behavior and temporal dependencies.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        temperature: Temperature scaling for attention weights (default: 1.0)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Linear transformations
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        self.head_importance = None
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        store_attention: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight storage.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
            store_attention: Whether to store attention weights for later analysis
            
        Returns:
            output: Transformed tensor of shape (batch_size, seq_len, d_model)
            attention_weights: Optional attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and split heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation: (B, H, L, D_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask for multi-head: (B, 1, L, L)
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # Expand for heads: (B, 1, L, L)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights if requested
        if store_attention:
            self.attention_weights = attention_weights.detach()
            self.head_importance = self._compute_head_importance(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (B, H, L, D_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final output projection
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _compute_head_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each attention head.
        
        Args:
            attention_weights: Attention weights (B, H, L, L)
            
        Returns:
            head_importance: Importance scores (H,)
        """
        # Compute entropy for each head (lower entropy = more focused)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1)
        
        # Average entropy across batch and sequence
        avg_entropy = entropy.mean(dim=(0, 2))  # (H,)
        
        # Convert to importance (inverse of entropy, normalized)
        importance = 1.0 / (avg_entropy + 1e-8)
        importance = importance / importance.sum()
        
        return importance
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get stored attention weights."""
        return self.attention_weights
    
    def get_head_importance(self) -> Optional[torch.Tensor]:
        """Get computed head importance scores."""
        return self.head_importance
    
    def aggregate_attention_heads(
        self, 
        method: str = "mean",
        head_indices: Optional[List[int]] = None
    ) -> Optional[torch.Tensor]:
        """
        Aggregate attention weights across heads.
        
        Args:
            method: Aggregation method ("mean", "max", "weighted", "top_k")
            head_indices: Specific head indices to aggregate (optional)
            
        Returns:
            aggregated_attention: Aggregated attention weights (B, L, L)
        """
        if self.attention_weights is None:
            return None
        
        attention = self.attention_weights  # (B, H, L, L)
        
        if head_indices is not None:
            attention = attention[:, head_indices, :, :]
        
        if method == "mean":
            return attention.mean(dim=1)
        
        elif method == "max":
            return attention.max(dim=1)[0]
        
        elif method == "weighted":
            if self.head_importance is None:
                return attention.mean(dim=1)
            
            # Weight by head importance
            importance = self.head_importance
            if head_indices is not None:
                importance = importance[head_indices]
            
            importance = importance.view(1, -1, 1, 1)  # (1, H, 1, 1)
            weighted_attention = attention * importance
            return weighted_attention.sum(dim=1)
        
        elif method == "top_k":
            # Use top-k most important heads
            k = min(4, attention.shape[1])  # Top 4 heads or all if fewer
            if self.head_importance is not None:
                _, top_indices = self.head_importance.topk(k)
                return attention[:, top_indices, :, :].mean(dim=1)
            else:
                return attention[:, :k, :, :].mean(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class AttentionVisualizer:
    """
    Utility class for visualizing attention patterns.
    
    Provides methods to create heatmaps, temporal attention plots,
    and head importance visualizations.
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize visualizer.
        
        Args:
            feature_names: Names of input features for labeling
        """
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(7)]
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        title: str = "Attention Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights (L, L) or (B, L, L)
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Handle batch dimension
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]  # Take first sample
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            annot=False,
            cmap='Blues',
            cbar=True,
            square=True,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_attention(
        self,
        attention_weights: torch.Tensor,
        query_position: int,
        title: str = "Temporal Attention Pattern",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention weights for a specific query position over time.
        
        Args:
            attention_weights: Attention weights (L, L) or (B, L, L)
            query_position: Query position to analyze
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Handle batch dimension
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Extract attention for specific query
        query_attention = attention_weights[query_position, :]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot attention weights
        positions = np.arange(len(query_attention))
        ax.plot(positions, query_attention, 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(positions, query_attention, alpha=0.3)
        
        # Highlight current position
        ax.axvline(x=query_position, color='red', linestyle='--', alpha=0.7, 
                  label=f'Query Position {query_position}')
        
        ax.set_title(f"{title} (Query Position {query_position})", fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_head_importance(
        self,
        head_importance: torch.Tensor,
        title: str = "Attention Head Importance",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot importance scores for attention heads.
        
        Args:
            head_importance: Head importance scores (H,)
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Convert to numpy
        if isinstance(head_importance, torch.Tensor):
            head_importance = head_importance.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        head_indices = np.arange(len(head_importance))
        bars = ax.bar(head_indices, head_importance, color='skyblue', alpha=0.7)
        
        # Highlight most important heads
        max_importance = head_importance.max()
        for i, (bar, importance) in enumerate(zip(bars, head_importance)):
            if importance > max_importance * 0.8:  # Top 20% importance
                bar.set_color('orange')
                bar.set_alpha(0.9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Attention Head', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_xticks(head_indices)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multi_head_comparison(
        self,
        attention_weights: torch.Tensor,
        head_indices: List[int],
        query_position: int,
        title: str = "Multi-Head Attention Comparison",
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare attention patterns across multiple heads.
        
        Args:
            attention_weights: Attention weights (B, H, L, L)
            head_indices: List of head indices to compare
            query_position: Query position to analyze
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Handle batch dimension
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Take first sample
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Create subplots
        n_heads = len(head_indices)
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each head
        for idx, head_idx in enumerate(head_indices):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Extract attention for this head and query
            head_attention = attention_weights[head_idx, query_position, :]
            positions = np.arange(len(head_attention))
            
            ax.plot(positions, head_attention, 'b-', linewidth=2, marker='o', markersize=3)
            ax.fill_between(positions, head_attention, alpha=0.3)
            ax.axvline(x=query_position, color='red', linestyle='--', alpha=0.7)
            
            ax.set_title(f'Head {head_idx}', fontsize=12)
            ax.set_xlabel('Key Position', fontsize=10)
            ax.set_ylabel('Attention Weight', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f"{title} (Query Position {query_position})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class AttentionAnalyzer:
    """
    Advanced analysis tools for attention patterns.
    
    Provides methods to analyze attention patterns, detect anomalies,
    and extract insights about model behavior.
    """
    
    def __init__(self):
        """Initialize attention analyzer."""
        pass
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention distributions.
        
        Args:
            attention_weights: Attention weights (B, H, L, L) or (B, L, L)
            
        Returns:
            entropy: Entropy values
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_attention = torch.log(attention_weights + epsilon)
        entropy = -(attention_weights * log_attention).sum(dim=-1)
        
        return entropy
    
    def detect_attention_anomalies(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Detect anomalous attention patterns.
        
        Args:
            attention_weights: Attention weights (B, H, L, L)
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            anomalies: Dictionary containing anomaly information
        """
        # Compute attention statistics
        entropy = self.compute_attention_entropy(attention_weights)
        max_attention = attention_weights.max(dim=-1)[0]
        
        # Compute z-scores
        entropy_mean = entropy.mean()
        entropy_std = entropy.std()
        entropy_zscore = (entropy - entropy_mean) / (entropy_std + 1e-8)
        
        max_attention_mean = max_attention.mean()
        max_attention_std = max_attention.std()
        max_attention_zscore = (max_attention - max_attention_mean) / (max_attention_std + 1e-8)
        
        # Detect anomalies
        entropy_anomalies = torch.abs(entropy_zscore) > threshold
        attention_anomalies = torch.abs(max_attention_zscore) > threshold
        
        return {
            'entropy_anomalies': entropy_anomalies,
            'attention_anomalies': attention_anomalies,
            'entropy_zscore': entropy_zscore,
            'max_attention_zscore': max_attention_zscore
        }
    
    def analyze_temporal_patterns(
        self,
        attention_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze temporal patterns in attention.
        
        Args:
            attention_weights: Attention weights (B, H, L, L)
            
        Returns:
            patterns: Dictionary of temporal pattern metrics
        """
        # Average across batch and heads
        if attention_weights.dim() == 4:
            avg_attention = attention_weights.mean(dim=(0, 1))  # (L, L)
        else:
            avg_attention = attention_weights.mean(dim=0)  # (L, L)
        
        # Compute metrics
        seq_len = avg_attention.shape[0]
        
        # Locality: How much attention focuses on nearby positions
        locality_scores = []
        for i in range(seq_len):
            distances = torch.abs(torch.arange(seq_len, device=avg_attention.device) - i)
            weighted_distance = (avg_attention[i] * distances).sum()
            locality_scores.append(weighted_distance.item())
        
        avg_locality = np.mean(locality_scores)
        
        # Recency bias: How much attention focuses on recent positions
        recency_weights = []
        for i in range(seq_len):
            recent_attention = avg_attention[i, max(0, i-10):i+1].sum()
            total_attention = avg_attention[i].sum()
            recency_weights.append((recent_attention / total_attention).item())
        
        avg_recency = np.mean(recency_weights)
        
        # Diagonal dominance: How much attention focuses on the diagonal
        diagonal_attention = torch.diag(avg_attention).sum().item()
        total_attention = avg_attention.sum().item()
        diagonal_ratio = diagonal_attention / total_attention
        
        return {
            'average_locality': avg_locality,
            'recency_bias': avg_recency,
            'diagonal_dominance': diagonal_ratio,
            'attention_spread': avg_attention.std().item()
        }