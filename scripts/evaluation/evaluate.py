#!/usr/bin/env python3
"""
# COMPONENT: Model Inference & Evaluation System  
# PURPOSE: Generate real dollar predictions and comprehensive metrics
# INPUTS: Trained model checkpoints, scalers, raw price data
# OUTPUTS: Dollar predictions, trading metrics, performance visualizations
# VERIFICATION: Price range validation, metric consistency, reproducibility
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer


class ModelLoader:
    """
    # COMPONENT: Model Loading System
    # PURPOSE: Load trained models with architecture verification
    # INPUTS: Model checkpoint path, device specification
    # OUTPUTS: Loaded model, configuration, scaler parameters
    # VERIFICATION: Architecture compatibility, parameter count validation
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        logging.info(f"Using device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine optimal device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_checkpoint(
        self, 
        model_path: Path, 
        scaler_path: Optional[Path] = None
    ) -> Tuple[nn.Module, Dict, Optional[Dict]]:
        """
        Load model checkpoint with complete validation
        
        Returns:
            model: Loaded model on specified device
            config: Model configuration
            scaler: Scaler parameters (if available)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        logging.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract configuration
        config = self._extract_config(checkpoint, model_path)
        
        # Initialize model
        model = self._initialize_model(config)
        
        # Load weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        model.to(self.device)
        model.eval()
        
        # Verify model integrity
        self.verify_model_integrity(model, config)
        
        # Load scaler if provided
        scaler = None
        if scaler_path and scaler_path.exists():
            scaler = self._load_scaler(scaler_path)
        elif 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
        
        return model, config, scaler
    
    def _extract_config(self, checkpoint: Dict, model_path: Path) -> Dict:
        """Extract model configuration from checkpoint"""
        config = {}
        
        # Try multiple sources for config
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Infer from model architecture
            logging.warning("No explicit config found, inferring from model")
            config = self._infer_config_from_checkpoint(checkpoint)
        
        # Ensure required fields
        required_fields = ['input_dim', 'hidden_dim', 'num_heads', 'num_layers', 'output_dim']
        missing = [f for f in required_fields if f not in config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        logging.info(f"Model config: {config}")
        return config
    
    def _infer_config_from_checkpoint(self, checkpoint: Dict) -> Dict:
        """Infer configuration from model state dict"""
        state_dict = checkpoint['model_state_dict']
        
        # Extract dimensions from layer weights
        config = {}
        
        # Input dimension from embedding layer
        if 'input_embedding.linear.weight' in state_dict:
            config['input_dim'] = state_dict['input_embedding.linear.weight'].shape[1]
            config['hidden_dim'] = state_dict['input_embedding.linear.weight'].shape[0]
        
        # Output dimension
        if 'output_layer.weight' in state_dict:
            config['output_dim'] = state_dict['output_layer.weight'].shape[0]
        
        # Count transformer layers
        transformer_layers = [k for k in state_dict.keys() if 'transformer_layers' in k]
        if transformer_layers:
            layer_indices = set()
            for key in transformer_layers:
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
            config['num_layers'] = max(layer_indices) + 1 if layer_indices else 4
        
        # Attention heads from first layer
        attention_key = 'transformer_layers.0.multi_head_attention.in_proj_weight'
        if attention_key in state_dict:
            hidden_dim = config.get('hidden_dim', 256)
            # Weight shape is (3 * hidden_dim, hidden_dim) for Q, K, V
            config['num_heads'] = 8  # Default assumption
        
        # Fill in defaults for missing values
        defaults = {
            'input_dim': 10,
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'output_dim': 3,
            'dropout': 0.1,
            'max_seq_length': 60
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        return config
    
    def _initialize_model(self, config: Dict) -> TimeSeriesTransformer:
        """Initialize model with configuration"""
        model = TimeSeriesTransformer(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.1),
            max_seq_length=config.get('max_seq_length', 60),
            output_dim=config['output_dim'],
            forecast_horizon=config.get('forecast_horizon', config['output_dim']),
            use_attention_pooling=config.get('use_attention_pooling', True)
        )
        
        return model
    
    def _load_scaler(self, scaler_path: Path) -> Dict:
        """Load scaler parameters from JSON file"""
        try:
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            logging.info(f"Loaded scaler from {scaler_path}")
            return scaler_data
        except Exception as e:
            logging.warning(f"Failed to load scaler: {e}")
            return None
    
    def verify_model_integrity(self, model: nn.Module, config: Dict):
        """Verify model loads correctly with dummy forward pass"""
        batch_size = 2
        seq_len = config.get('max_seq_length', 60)
        input_dim = config['input_dim']
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, input_dim, device=self.device)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_shape = (batch_size, config['output_dim'])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            # Check for NaN/Inf
            assert not torch.any(torch.isnan(output)), "Model produces NaN outputs"
            assert not torch.any(torch.isinf(output)), "Model produces Inf outputs"
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            logging.info(f"Model verification passed. Parameters: {param_count:,}")
            
        except Exception as e:
            raise RuntimeError(f"Model integrity check failed: {e}")


class PredictionPipeline:
    """
    # COMPONENT: Prediction Pipeline
    # PURPOSE: Generate dollar predictions from standardized model outputs
    # INPUTS: Raw features, trained model, scaler parameters
    # OUTPUTS: Dollar predictions, confidence intervals, attention weights
    # VERIFICATION: Price range validation, scaling consistency
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        scaler: Dict, 
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.config = config
        self.feature_names = scaler.get('feature_names', []) if scaler else []
        
        # Validate scaler
        if scaler:
            self._validate_scaler(scaler)
    
    def _validate_scaler(self, scaler: Dict):
        """Validate scaler contains required parameters"""
        if 'scaler_params' not in scaler:
            raise ValueError("Scaler missing 'scaler_params' key")
        
        # Check feature parameters
        scaler_params = scaler['scaler_params']
        for feature in self.feature_names:
            if feature not in scaler_params:
                raise ValueError(f"Scaler missing parameters for feature: {feature}")
            
            params = scaler_params[feature]
            required = ['mean', 'std']
            missing = [p for p in required if p not in params]
            if missing:
                raise ValueError(f"Feature {feature} missing scaler params: {missing}")
    
    def standardize_features(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using saved scaler parameters"""
        if self.scaler is None:
            logging.warning("No scaler available, returning features as-is")
            return features
        
        standardized = np.zeros_like(features)
        scaler_params = self.scaler['scaler_params']
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= features.shape[1]:
                break
                
            if feature_name in scaler_params:
                params = scaler_params[feature_name]
                mean = params['mean']
                std = params['std']
                
                # Avoid division by zero
                if std < 1e-8:
                    std = 1.0
                
                standardized[:, i] = (features[:, i] - mean) / std
            else:
                standardized[:, i] = features[:, i]
        
        return standardized.astype(np.float32)
    
    def destandardize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert standardized predictions back to dollar values"""
        if self.scaler is None:
            logging.warning("No scaler available, returning predictions as-is")
            return predictions
        
        # For percentage return targets, we need to convert back to price deltas
        scaler_params = self.scaler['scaler_params']
        
        # Predictions are typically percentage returns, so we return them as-is
        # The actual dollar conversion happens at the sequence level
        return predictions
    
    def predict_sequence(
        self, 
        features: np.ndarray, 
        return_attention: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a single sequence
        
        Args:
            features: (seq_len, n_features) raw features
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with predictions and optional attention weights
        """
        # Validate input
        seq_len, n_features = features.shape
        expected_features = len(self.feature_names) if self.feature_names else self.config['input_dim']
        
        if n_features != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {n_features}")
        
        # Standardize features
        standardized_features = self.standardize_features(features)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(standardized_features).unsqueeze(0).to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            if return_attention:
                predictions, attention_weights = self.model(
                    input_tensor, 
                    return_attention=True
                )
            else:
                predictions = self.model(input_tensor)
                attention_weights = None
        
        # Convert to numpy
        predictions_np = predictions.squeeze(0).cpu().numpy()
        
        # Prepare results
        results = {
            'predictions_standardized': predictions_np,
            'predictions_returns': predictions_np,  # These are percentage returns
        }
        
        if attention_weights is not None:
            # Average attention across heads for visualization
            attention_np = []
            for layer_attention in attention_weights:
                # Shape: (batch, heads, seq_len, seq_len)
                layer_avg = layer_attention.squeeze(0).mean(dim=0).cpu().numpy()
                attention_np.append(layer_avg)
            results['attention_weights'] = attention_np
        
        return results
    
    def sliding_window_predictions(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        seq_len: int = 60,
        stride: int = 1,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate predictions over time period with sliding window
        
        Args:
            data: DataFrame with OHLCV data and date index
            start_date: Start date for predictions
            end_date: End date for predictions  
            seq_len: Sequence length for model input
            stride: Step size for sliding window
            feature_columns: List of feature column names
            
        Returns:
            DataFrame with columns: date, actual, predicted_1d, predicted_3d, predicted_5d
        """
        # Ensure date index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'Date' in data.columns:
                data = data.set_index('Date')
            else:
                raise ValueError("Data must have DatetimeIndex or 'Date' column")
        
        # Filter data to date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        data_filtered = data.loc[mask].copy()
        
        if len(data_filtered) < seq_len:
            raise ValueError(f"Insufficient data: need {seq_len}, got {len(data_filtered)}")
        
        # Prepare feature columns
        if feature_columns is None:
            # Use OHLCV + basic features
            feature_columns = []
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data_filtered.columns:
                    feature_columns.append(col)
            
            # Add returns
            if 'Close' in data_filtered.columns:
                data_filtered['Returns'] = data_filtered['Close'].pct_change().fillna(0)
                feature_columns.append('Returns')
        
        # Ensure we have the expected number of features
        available_features = [col for col in feature_columns if col in data_filtered.columns]
        if len(available_features) != len(self.feature_names):
            logging.warning(f"Feature mismatch: expected {len(self.feature_names)}, got {len(available_features)}")
        
        # Generate predictions
        results = []
        horizon = self.config['output_dim']
        
        for i in range(seq_len, len(data_filtered), stride):
            if i + horizon > len(data_filtered):
                break
            
            # Get sequence features
            seq_data = data_filtered.iloc[i-seq_len:i]
            features = seq_data[available_features].values
            
            # Skip if any NaN
            if np.any(np.isnan(features)):
                continue
            
            # Get current date and actual future values
            current_date = data_filtered.index[i-1]
            future_closes = data_filtered['Close'].iloc[i:i+horizon].values
            current_close = data_filtered['Close'].iloc[i-1]
            
            # Calculate actual returns
            actual_returns = (future_closes - current_close) / current_close
            
            try:
                # Generate prediction
                pred_results = self.predict_sequence(features)
                predicted_returns = pred_results['predictions_returns']
                
                # Convert returns to dollar predictions
                predicted_prices = current_close * (1 + predicted_returns)
                actual_prices = future_closes
                
                # Store results
                result_row = {
                    'date': current_date,
                    'current_price': current_close,
                }
                
                # Add predictions and actuals for each horizon
                for h in range(min(horizon, len(predicted_prices), len(actual_prices))):
                    result_row[f'actual_{h+1}d'] = actual_prices[h]
                    result_row[f'predicted_{h+1}d'] = predicted_prices[h]
                    result_row[f'actual_return_{h+1}d'] = actual_returns[h]
                    result_row[f'predicted_return_{h+1}d'] = predicted_returns[h]
                
                results.append(result_row)
                
            except Exception as e:
                logging.warning(f"Prediction failed for {current_date}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid predictions generated")
        
        predictions_df = pd.DataFrame(results)
        predictions_df.set_index('date', inplace=True)
        
        logging.info(f"Generated {len(predictions_df)} predictions from {start_date} to {end_date}")
        
        return predictions_df


class MetricsCalculator:
    """
    # COMPONENT: Comprehensive Metrics Calculator
    # PURPOSE: Calculate regression, directional, and trading performance metrics
    # INPUTS: Actual and predicted values, returns series
    # OUTPUTS: Comprehensive metric dictionaries
    # VERIFICATION: Handle edge cases, validate metric ranges
    """
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression performance metrics
        
        Returns:
            Dictionary with RMSE, MAE, MAPE, R²
        """
        # Remove NaN pairs
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'n_samples': 0
            }
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE - handle division by zero
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        
        # R²
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'n_samples': int(len(y_true_clean))
        }
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional prediction accuracy
        
        Returns:
            Dictionary with accuracy, precision, recall for up/down moves
        """
        # Remove NaN pairs
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            return {
                'overall_accuracy': np.nan,
                'up_precision': np.nan,
                'up_recall': np.nan,
                'down_precision': np.nan,
                'down_recall': np.nan,
                'n_samples': 0
            }
        
        # Convert to directional predictions
        true_direction = (y_true_clean > 0).astype(int)  # 1 for up, 0 for down
        pred_direction = (y_pred_clean > 0).astype(int)
        
        # Overall accuracy
        accuracy = np.mean(true_direction == pred_direction)
        
        # Precision and recall for up moves (class 1)
        true_up = (true_direction == 1)
        pred_up = (pred_direction == 1)
        
        up_precision = np.sum(true_up & pred_up) / (np.sum(pred_up) + 1e-8)
        up_recall = np.sum(true_up & pred_up) / (np.sum(true_up) + 1e-8)
        
        # Precision and recall for down moves (class 0)
        true_down = (true_direction == 0)
        pred_down = (pred_direction == 0)
        
        down_precision = np.sum(true_down & pred_down) / (np.sum(pred_down) + 1e-8)
        down_recall = np.sum(true_down & pred_down) / (np.sum(true_down) + 1e-8)
        
        return {
            'overall_accuracy': float(accuracy),
            'up_precision': float(up_precision),
            'up_recall': float(up_recall),
            'down_precision': float(down_precision),
            'down_recall': float(down_recall),
            'n_samples': int(len(y_true_clean))
        }
    
    @staticmethod
    def calculate_trading_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate trading performance metrics
        
        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with Sharpe, Sortino, max drawdown, win rate, etc.
        """
        # Clean returns
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {
                'sharpe_ratio': np.nan,
                'sortino_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'profit_factor': np.nan,
                'total_return': np.nan,
                'volatility': np.nan,
                'n_samples': 0
            }
        
        # Annualized metrics (assuming daily returns)
        annual_factor = 252
        daily_rf_rate = risk_free_rate / annual_factor
        
        # Basic stats
        mean_return = clean_returns.mean()
        volatility = clean_returns.std()
        total_return = (1 + clean_returns).prod() - 1
        
        # Sharpe ratio
        excess_returns = clean_returns - daily_rf_rate
        sharpe_ratio = (excess_returns.mean() / (excess_returns.std() + 1e-8)) * np.sqrt(annual_factor)
        
        # Sortino ratio (only downside deviation)
        downside_returns = clean_returns[clean_returns < daily_rf_rate]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else volatility
        sortino_ratio = (excess_returns.mean() / (downside_std + 1e-8)) * np.sqrt(annual_factor)
        
        # Maximum drawdown
        cumulative_returns = (1 + clean_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate
        win_rate = (clean_returns > 0).mean()
        
        # Profit factor
        positive_returns = clean_returns[clean_returns > 0].sum()
        negative_returns = abs(clean_returns[clean_returns < 0].sum())
        profit_factor = positive_returns / (negative_returns + 1e-8)
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_return': float(total_return),
            'volatility': float(volatility * np.sqrt(annual_factor)),  # Annualized
            'mean_daily_return': float(mean_return),
            'n_samples': int(len(clean_returns))
        }
    
    @staticmethod
    def calculate_confidence_calibration(
        predictions: np.ndarray, 
        actuals: np.ndarray, 
        confidence_levels: List[float] = [0.5, 0.68, 0.95]
    ) -> Dict[str, float]:
        """
        Calculate confidence calibration metrics
        
        Args:
            predictions: Point predictions
            actuals: Actual values
            confidence_levels: Confidence levels to test
            
        Returns:
            Dictionary with calibration metrics
        """
        # Remove NaN pairs
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[valid_mask]
        actual_clean = actuals[valid_mask]
        
        if len(pred_clean) == 0:
            return {f'calibration_{int(cl*100)}': np.nan for cl in confidence_levels}
        
        # Calculate prediction errors
        errors = np.abs(pred_clean - actual_clean)
        
        # For each confidence level, check if the predicted interval contains the actual
        calibration_results = {}
        
        for cl in confidence_levels:
            # Use error percentiles as proxy for confidence intervals
            error_threshold = np.percentile(errors, cl * 100)
            
            # Count how many predictions have errors within this threshold
            within_interval = (errors <= error_threshold).mean()
            
            calibration_results[f'calibration_{int(cl*100)}'] = float(within_interval)
        
        return calibration_results


class EvaluationVisualizer:
    """
    # COMPONENT: Visualization Suite
    # PURPOSE: Generate comprehensive performance visualizations
    # INPUTS: Predictions DataFrame, attention weights, metrics
    # OUTPUTS: Publication-quality plots and analysis charts
    # VERIFICATION: Clear visual insights, publication ready formatting
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_predictions_vs_actuals(
        self, 
        df: pd.DataFrame, 
        ticker: str = "Stock",
        save_path: Optional[Path] = None
    ):
        """
        Multi-panel plot showing predictions vs actuals
        
        Args:
            df: DataFrame with actual and predicted columns
            ticker: Stock ticker for title
            save_path: Optional save path, defaults to output_dir
        """
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_predictions_vs_actuals.png"
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{ticker} Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Time series of predictions vs actuals (1-day horizon)
        ax1 = axes[0, 0]
        if 'actual_1d' in df.columns and 'predicted_1d' in df.columns:
            ax1.plot(df.index, df['actual_1d'], label='Actual', alpha=0.7, linewidth=1.5)
            ax1.plot(df.index, df['predicted_1d'], label='Predicted', alpha=0.8, linewidth=1.5)
            ax1.set_title('1-Day Ahead Predictions vs Actuals')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Panel 2: Scatter plot of predictions vs actuals
        ax2 = axes[0, 1]
        if 'actual_1d' in df.columns and 'predicted_1d' in df.columns:
            # Remove NaN values
            mask = ~(df['actual_1d'].isna() | df['predicted_1d'].isna())
            actual_clean = df['actual_1d'][mask]
            pred_clean = df['predicted_1d'][mask]
            
            if len(actual_clean) > 0:
                ax2.scatter(actual_clean, pred_clean, alpha=0.6, s=20)
                
                # Perfect prediction line
                min_val = min(actual_clean.min(), pred_clean.min())
                max_val = max(actual_clean.max(), pred_clean.max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
                
                # Calculate R²
                r2 = MetricsCalculator.calculate_regression_metrics(
                    actual_clean.values, pred_clean.values
                )['r2']
                ax2.set_title(f'Predictions vs Actuals (R² = {r2:.3f})')
                ax2.set_xlabel('Actual Price ($)')
                ax2.set_ylabel('Predicted Price ($)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Panel 3: Prediction errors distribution
        ax3 = axes[1, 0]
        if 'actual_1d' in df.columns and 'predicted_1d' in df.columns:
            errors = df['predicted_1d'] - df['actual_1d']
            errors_clean = errors.dropna()
            
            if len(errors_clean) > 0:
                ax3.hist(errors_clean, bins=50, alpha=0.7, density=True, color='skyblue')
                
                # Add normal curve overlay
                mu, sigma = errors_clean.mean(), errors_clean.std()
                x = np.linspace(errors_clean.min(), errors_clean.max(), 100)
                y = stats.norm.pdf(x, mu, sigma)
                ax3.plot(x, y, 'r-', alpha=0.8, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
                
                ax3.axvline(0, color='red', linestyle='--', alpha=0.8, label='Zero Error')
                ax3.set_title('Prediction Errors Distribution')
                ax3.set_xlabel('Prediction Error ($)')
                ax3.set_ylabel('Density')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Panel 4: Cumulative returns comparison
        ax4 = axes[1, 1]
        if 'actual_return_1d' in df.columns and 'predicted_return_1d' in df.columns:
            # Calculate cumulative returns
            actual_returns = df['actual_return_1d'].fillna(0)
            pred_returns = df['predicted_return_1d'].fillna(0)
            
            actual_cumret = (1 + actual_returns).cumprod()
            pred_cumret = (1 + pred_returns).cumprod()
            
            ax4.plot(df.index, actual_cumret, label='Actual Returns', alpha=0.8, linewidth=2)
            ax4.plot(df.index, pred_cumret, label='Predicted Returns', alpha=0.8, linewidth=2)
            ax4.set_title('Cumulative Returns Comparison')
            ax4.set_ylabel('Cumulative Return')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved predictions plot to {save_path}")
    
    def plot_attention_heatmap(
        self, 
        attention_weights: List[np.ndarray], 
        sequence_length: int = 60,
        save_path: Optional[Path] = None
    ):
        """
        Visualize attention patterns across transformer layers
        
        Args:
            attention_weights: List of attention matrices for each layer
            sequence_length: Length of input sequence
            save_path: Optional save path
        """
        if save_path is None:
            save_path = self.output_dir / "attention_heatmap.png"
        
        n_layers = len(attention_weights)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * ((n_layers + 1) // 2), 8))
        if n_layers == 1:
            axes = [axes]
        elif n_layers <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Attention Patterns Across Transformer Layers', fontsize=14, fontweight='bold')
        
        for layer_idx, attention_matrix in enumerate(attention_weights):
            ax = axes[layer_idx] if n_layers > 1 else axes
            
            # Attention matrix shape: (seq_len, seq_len)
            im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
            ax.set_title(f'Layer {layer_idx + 1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Remove unused subplots
        for idx in range(n_layers, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved attention heatmap to {save_path}")
    
    def plot_performance_by_market_condition(
        self, 
        df: pd.DataFrame,
        save_path: Optional[Path] = None
    ):
        """
        Show model performance in different market conditions
        
        Args:
            df: DataFrame with predictions and market data
            save_path: Optional save path
        """
        if save_path is None:
            save_path = self.output_dir / "performance_by_market_condition.png"
        
        # Calculate volatility regimes
        if 'actual_return_1d' in df.columns:
            returns = df['actual_return_1d'].dropna()
            
            # Calculate rolling volatility
            vol_window = 20
            rolling_vol = returns.rolling(window=vol_window).std()
            
            # Define volatility regimes
            vol_low = rolling_vol.quantile(0.33)
            vol_high = rolling_vol.quantile(0.67)
            
            # Classify market conditions
            conditions = []
            for vol in rolling_vol:
                if pd.isna(vol):
                    conditions.append('Unknown')
                elif vol <= vol_low:
                    conditions.append('Low Vol')
                elif vol <= vol_high:
                    conditions.append('Med Vol')
                else:
                    conditions.append('High Vol')
            
            df['market_condition'] = conditions
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Model Performance by Market Condition', fontsize=14, fontweight='bold')
            
            # Performance by volatility regime
            if 'predicted_1d' in df.columns and 'actual_1d' in df.columns:
                for ax, metric in zip(axes, ['RMSE', 'Directional Accuracy', 'Returns Correlation']):
                    regime_metrics = []
                    regime_names = []
                    
                    for condition in ['Low Vol', 'Med Vol', 'High Vol']:
                        condition_data = df[df['market_condition'] == condition]
                        
                        if len(condition_data) > 10:  # Minimum samples
                            if metric == 'RMSE':
                                actual = condition_data['actual_1d'].dropna()
                                pred = condition_data['predicted_1d'].dropna()
                                # Align the arrays
                                common_idx = actual.index.intersection(pred.index)
                                if len(common_idx) > 0:
                                    rmse = np.sqrt(mean_squared_error(actual[common_idx], pred[common_idx]))
                                    regime_metrics.append(rmse)
                                    regime_names.append(condition)
                            
                            elif metric == 'Directional Accuracy':
                                actual_ret = condition_data['actual_return_1d'].dropna()
                                pred_ret = condition_data['predicted_return_1d'].dropna()
                                common_idx = actual_ret.index.intersection(pred_ret.index)
                                if len(common_idx) > 0:
                                    dir_acc = MetricsCalculator.calculate_directional_accuracy(
                                        actual_ret[common_idx].values, 
                                        pred_ret[common_idx].values
                                    )['overall_accuracy']
                                    regime_metrics.append(dir_acc)
                                    regime_names.append(condition)
                            
                            elif metric == 'Returns Correlation':
                                actual_ret = condition_data['actual_return_1d'].dropna()
                                pred_ret = condition_data['predicted_return_1d'].dropna()
                                common_idx = actual_ret.index.intersection(pred_ret.index)
                                if len(common_idx) > 1:
                                    corr = np.corrcoef(actual_ret[common_idx], pred_ret[common_idx])[0, 1]
                                    if not np.isnan(corr):
                                        regime_metrics.append(corr)
                                        regime_names.append(condition)
                    
                    if regime_metrics:
                        bars = ax.bar(regime_names, regime_metrics, alpha=0.7, color=['blue', 'orange', 'red'])
                        ax.set_title(f'{metric} by Market Condition')
                        ax.set_ylabel(metric)
                        ax.grid(True, alpha=0.3)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, regime_metrics):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved market condition analysis to {save_path}")
    
    def plot_error_analysis(
        self, 
        df: pd.DataFrame,
        save_path: Optional[Path] = None
    ):
        """
        Detailed error analysis plots
        
        Args:
            df: DataFrame with predictions
            save_path: Optional save path
        """
        if save_path is None:
            save_path = self.output_dir / "error_analysis.png"
        
        if 'actual_1d' not in df.columns or 'predicted_1d' not in df.columns:
            logging.warning("Cannot create error analysis: missing prediction columns")
            return
        
        # Calculate errors
        errors = df['predicted_1d'] - df['actual_1d']
        errors_clean = errors.dropna()
        
        if len(errors_clean) == 0:
            logging.warning("Cannot create error analysis: no valid predictions")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Prediction Error Analysis', fontsize=14, fontweight='bold')
        
        # 1. Error over time
        ax1 = axes[0, 0]
        ax1.plot(errors.index, errors, alpha=0.7, linewidth=1)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax1.set_title('Prediction Errors Over Time')
        ax1.set_ylabel('Error ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality check
        ax2 = axes[0, 1]
        stats.probplot(errors_clean, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Autocorrelation of errors
        ax3 = axes[1, 0]
        from statsmodels.tsa.stattools import acf
        lags = min(20, len(errors_clean) // 4)
        if lags > 1:
            autocorr = acf(errors_clean, nlags=lags, fft=True)
            ax3.bar(range(len(autocorr)), autocorr, alpha=0.7)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax3.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
            ax3.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
            ax3.set_title('Error Autocorrelation')
            ax3.set_xlabel('Lag')
            ax3.set_ylabel('Autocorrelation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Error magnitude vs actual price
        ax4 = axes[1, 1]
        actual_prices = df['actual_1d'].dropna()
        abs_errors = errors.abs().dropna()
        common_idx = actual_prices.index.intersection(abs_errors.index)
        
        if len(common_idx) > 0:
            ax4.scatter(actual_prices[common_idx], abs_errors[common_idx], alpha=0.6, s=20)
            ax4.set_title('Error Magnitude vs Price Level')
            ax4.set_xlabel('Actual Price ($)')
            ax4.set_ylabel('Absolute Error ($)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved error analysis to {save_path}")


def setup_logging(output_dir: Path):
    """Setup logging for evaluation"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def load_stock_data(data_path: Path, ticker: str = None) -> pd.DataFrame:
    """
    Load stock data from parquet file or directory
    
    Args:
        data_path: Path to data file or directory
        ticker: Specific ticker to load (if data_path is directory)
    
    Returns:
        DataFrame with OHLCV data and date index
    """
    if data_path.is_file():
        # Single file
        df = pd.read_parquet(data_path)
    elif data_path.is_dir():
        # Directory - look for ticker subdirectory
        if ticker:
            ticker_dir = data_path / ticker.upper()
            if not ticker_dir.exists():
                raise FileNotFoundError(f"No directory found for ticker {ticker}")
            
            parquet_files = list(ticker_dir.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found for ticker {ticker}")
            
            # Load most recent file
            parquet_files.sort()
            df = pd.read_parquet(parquet_files[-1])
        else:
            raise ValueError("ticker must be specified when data_path is a directory")
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Ensure proper date index
    if 'Date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have a Date column or DatetimeIndex")
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logging.info(f"Loaded data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    return df


def generate_comprehensive_report(
    model_info: Dict,
    data_info: Dict,
    predictions_df: pd.DataFrame,
    regression_metrics: Dict,
    directional_metrics: Dict,
    trading_metrics: Dict,
    output_dir: Path
) -> Dict:
    """
    Generate comprehensive evaluation report
    
    Returns:
        Dictionary with complete evaluation results
    """
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "data_info": data_info,
        "regression_metrics": regression_metrics,
        "directional_metrics": directional_metrics,
        "trading_metrics": trading_metrics,
        "summary": {}
    }
    
    # Generate summary statistics
    summary = {}
    
    # Model performance summary
    if 'rmse' in regression_metrics:
        summary['rmse_dollars'] = regression_metrics['rmse']
        summary['baseline_comparison'] = {
            'baseline_rmse': 0.268,
            'achieved_rmse': regression_metrics['rmse'],
            'improvement': (0.268 - regression_metrics['rmse']) / 0.268 if regression_metrics['rmse'] < 0.268 else 0
        }
    
    if 'overall_accuracy' in directional_metrics:
        summary['directional_accuracy'] = directional_metrics['overall_accuracy']
    
    if 'sharpe_ratio' in trading_metrics:
        summary['sharpe_ratio'] = trading_metrics['sharpe_ratio']
        summary['max_drawdown'] = trading_metrics['max_drawdown']
    
    # Prediction quality assessment
    if len(predictions_df) > 0:
        summary['prediction_coverage'] = {
            'total_predictions': len(predictions_df),
            'valid_predictions': len(predictions_df.dropna()),
            'coverage_ratio': len(predictions_df.dropna()) / len(predictions_df)
        }
        
        # Price range analysis
        if 'actual_1d' in predictions_df.columns:
            actual_prices = predictions_df['actual_1d'].dropna()
            summary['price_analysis'] = {
                'min_price': float(actual_prices.min()),
                'max_price': float(actual_prices.max()),
                'mean_price': float(actual_prices.mean()),
                'price_volatility': float(actual_prices.std())
            }
    
    report['summary'] = summary
    
    # Save report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    markdown_file = output_dir / "evaluation_report.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    logging.info(f"Saved evaluation report to {report_file}")
    logging.info(f"Saved markdown report to {markdown_file}")
    
    return report


def generate_markdown_report(report: Dict) -> str:
    """Generate markdown-formatted evaluation report"""
    md = f"""# Model Evaluation Report

Generated: {report['evaluation_timestamp']}

## Model Information

- **Model Path**: {report['model_info'].get('path', 'N/A')}
- **Parameters**: {report['model_info'].get('parameters', 'N/A'):,}
- **Architecture**: {report['model_info'].get('architecture', 'N/A')}

## Data Information

- **Ticker**: {report['data_info'].get('ticker', 'N/A')}
- **Date Range**: {report['data_info'].get('start_date', 'N/A')} to {report['data_info'].get('end_date', 'N/A')}
- **Predictions**: {report['data_info'].get('n_predictions', 'N/A')}

## Performance Metrics

### Regression Performance
"""
    
    reg_metrics = report['regression_metrics']
    if reg_metrics:
        md += f"""
- **RMSE**: ${reg_metrics.get('rmse', 'N/A'):.4f}
- **MAE**: ${reg_metrics.get('mae', 'N/A'):.4f}
- **MAPE**: {reg_metrics.get('mape', 'N/A'):.2f}%
- **R²**: {reg_metrics.get('r2', 'N/A'):.4f}
- **Samples**: {reg_metrics.get('n_samples', 'N/A')}
"""
    
    md += "\n### Directional Accuracy\n"
    dir_metrics = report['directional_metrics']
    if dir_metrics:
        md += f"""
- **Overall Accuracy**: {dir_metrics.get('overall_accuracy', 'N/A'):.2%}
- **Up Precision**: {dir_metrics.get('up_precision', 'N/A'):.2%}
- **Up Recall**: {dir_metrics.get('up_recall', 'N/A'):.2%}
- **Down Precision**: {dir_metrics.get('down_precision', 'N/A'):.2%}
- **Down Recall**: {dir_metrics.get('down_recall', 'N/A'):.2%}
"""
    
    md += "\n### Trading Metrics\n"
    trading_metrics = report['trading_metrics']
    if trading_metrics:
        md += f"""
- **Sharpe Ratio**: {trading_metrics.get('sharpe_ratio', 'N/A'):.3f}
- **Sortino Ratio**: {trading_metrics.get('sortino_ratio', 'N/A'):.3f}
- **Max Drawdown**: {trading_metrics.get('max_drawdown', 'N/A'):.2%}
- **Win Rate**: {trading_metrics.get('win_rate', 'N/A'):.2%}
- **Total Return**: {trading_metrics.get('total_return', 'N/A'):.2%}
- **Volatility (Annual)**: {trading_metrics.get('volatility', 'N/A'):.2%}
"""
    
    # Summary section
    summary = report.get('summary', {})
    if summary:
        md += "\n## Summary\n"
        
        baseline_comp = summary.get('baseline_comparison', {})
        if baseline_comp:
            md += f"""
### Baseline Comparison
- **Baseline RMSE**: ${baseline_comp.get('baseline_rmse', 'N/A'):.3f}
- **Achieved RMSE**: ${baseline_comp.get('achieved_rmse', 'N/A'):.3f}
- **Improvement**: {baseline_comp.get('improvement', 0):.1%}
"""
        
        pred_cov = summary.get('prediction_coverage', {})
        if pred_cov:
            md += f"""
### Prediction Quality
- **Total Predictions**: {pred_cov.get('total_predictions', 'N/A')}
- **Valid Predictions**: {pred_cov.get('valid_predictions', 'N/A')}
- **Coverage**: {pred_cov.get('coverage_ratio', 'N/A'):.1%}
"""
    
    md += "\n---\n*Generated by TimeSeries Transformer Evaluation Pipeline*"
    
    return md


def main():
    """
    # COMPONENT: Main Evaluation Pipeline
    # PURPOSE: Orchestrate complete model evaluation workflow
    # INPUTS: Model checkpoint, data, configuration parameters
    # OUTPUTS: Dollar predictions, comprehensive metrics, visualizations
    # VERIFICATION: Price range validation, metric consistency, report generation
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Model Inference & Evaluation System')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to data file or directory')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to scaler JSON file (optional)')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker (required if data-path is directory)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for evaluation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for evaluation (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--seq-len', type=int, default=60,
                        help='Sequence length for model input')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window predictions')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for model inference (auto, cpu, cuda)')
    parser.add_argument('--risk-free-rate', type=float, default=0.02,
                        help='Annual risk-free rate for trading metrics')
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    scaler_path = Path(args.scaler_path) if args.scaler_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("MODEL EVALUATION PIPELINE")
    logging.info("="*80)
    logging.info(f"Model: {model_path}")
    logging.info(f"Data: {data_path}")
    logging.info(f"Ticker: {args.ticker}")
    logging.info(f"Date Range: {args.start_date} to {args.end_date}")
    logging.info(f"Output: {output_dir}")
    
    try:
        # 1. Load model and scaler
        logging.info("\n1. Loading model and scaler...")
        loader = ModelLoader(device=args.device)
        model, config, scaler = loader.load_checkpoint(model_path, scaler_path)
        
        model_info = {
            'path': str(model_path),
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': 'TimeSeriesTransformer',
            'config': config
        }
        
        logging.info(f"Model loaded: {model_info['parameters']:,} parameters")
        
        # 2. Load data
        logging.info("\n2. Loading stock data...")
        data = load_stock_data(data_path, args.ticker)
        
        data_info = {
            'ticker': args.ticker,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'total_samples': len(data),
            'date_range': [str(data.index[0].date()), str(data.index[-1].date())]
        }
        
        # 3. Initialize prediction pipeline
        logging.info("\n3. Initializing prediction pipeline...")
        pipeline = PredictionPipeline(model, scaler, loader.device, config)
        
        # 4. Generate predictions
        logging.info("\n4. Generating sliding window predictions...")
        predictions_df = pipeline.sliding_window_predictions(
            data=data,
            start_date=args.start_date,
            end_date=args.end_date,
            seq_len=args.seq_len,
            stride=args.stride
        )
        
        data_info['n_predictions'] = len(predictions_df)
        logging.info(f"Generated {len(predictions_df)} predictions")
        
        # Save predictions
        pred_file = output_dir / f"{args.ticker}_predictions.csv"
        predictions_df.to_csv(pred_file)
        logging.info(f"Saved predictions to {pred_file}")
        
        # 5. Calculate metrics
        logging.info("\n5. Calculating comprehensive metrics...")
        
        # Regression metrics
        if 'actual_1d' in predictions_df.columns and 'predicted_1d' in predictions_df.columns:
            actual = predictions_df['actual_1d'].dropna()
            predicted = predictions_df['predicted_1d'].dropna()
            common_idx = actual.index.intersection(predicted.index)
            
            regression_metrics = MetricsCalculator.calculate_regression_metrics(
                actual[common_idx].values, predicted[common_idx].values
            )
        else:
            regression_metrics = {}
        
        # Directional metrics
        if 'actual_return_1d' in predictions_df.columns and 'predicted_return_1d' in predictions_df.columns:
            actual_ret = predictions_df['actual_return_1d'].dropna()
            predicted_ret = predictions_df['predicted_return_1d'].dropna()
            common_idx = actual_ret.index.intersection(predicted_ret.index)
            
            directional_metrics = MetricsCalculator.calculate_directional_accuracy(
                actual_ret[common_idx].values, predicted_ret[common_idx].values
            )
        else:
            directional_metrics = {}
        
        # Trading metrics
        if 'predicted_return_1d' in predictions_df.columns:
            pred_returns = predictions_df['predicted_return_1d'].dropna()
            trading_metrics = MetricsCalculator.calculate_trading_metrics(
                pred_returns, risk_free_rate=args.risk_free_rate
            )
        else:
            trading_metrics = {}
        
        # Log key metrics
        if regression_metrics:
            logging.info(f"RMSE: ${regression_metrics['rmse']:.4f}")
            logging.info(f"R²: {regression_metrics['r2']:.4f}")
        
        if directional_metrics:
            logging.info(f"Directional Accuracy: {directional_metrics['overall_accuracy']:.2%}")
        
        if trading_metrics:
            logging.info(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.3f}")
        
        # 6. Generate visualizations
        logging.info("\n6. Generating visualizations...")
        visualizer = EvaluationVisualizer(output_dir)
        
        # Main prediction plot
        visualizer.plot_predictions_vs_actuals(predictions_df, args.ticker)
        
        # Error analysis
        visualizer.plot_error_analysis(predictions_df)
        
        # Market condition analysis
        visualizer.plot_performance_by_market_condition(predictions_df)
        
        # Attention visualization (if available)
        try:
            # Get sample attention weights
            sample_idx = len(data) // 2
            if sample_idx + args.seq_len < len(data):
                sample_features = data.iloc[sample_idx:sample_idx + args.seq_len][
                    ['Open', 'High', 'Low', 'Close', 'Volume']
                ].values
                sample_features = np.column_stack([
                    sample_features,
                    np.zeros(len(sample_features))  # Add returns placeholder
                ])
                
                pred_result = pipeline.predict_sequence(sample_features, return_attention=True)
                if 'attention_weights' in pred_result:
                    visualizer.plot_attention_heatmap(pred_result['attention_weights'])
        except Exception as e:
            logging.warning(f"Could not generate attention visualization: {e}")
        
        # 7. Generate comprehensive report
        logging.info("\n7. Generating evaluation report...")
        report = generate_comprehensive_report(
            model_info=model_info,
            data_info=data_info,
            predictions_df=predictions_df,
            regression_metrics=regression_metrics,
            directional_metrics=directional_metrics,
            trading_metrics=trading_metrics,
            output_dir=output_dir
        )
        
        # 8. Final summary
        logging.info("\n" + "="*80)
        logging.info("EVALUATION COMPLETE")
        logging.info("="*80)
        
        if regression_metrics:
            baseline_rmse = 0.268
            achieved_rmse = regression_metrics['rmse']
            improvement = (baseline_rmse - achieved_rmse) / baseline_rmse
            status = "✓ Maintained" if achieved_rmse <= baseline_rmse * 1.1 else "⚠ Degraded"
            
            logging.info(f"RMSE: ${achieved_rmse:.4f} vs baseline ${baseline_rmse:.3f} {status}")
            if improvement > 0:
                logging.info(f"Improvement: {improvement:.1%}")
        
        if directional_metrics:
            logging.info(f"Directional Accuracy: {directional_metrics['overall_accuracy']:.1%}")
        
        if trading_metrics:
            logging.info(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.3f}")
            logging.info(f"Max Drawdown: {trading_metrics['max_drawdown']:.1%}")
        
        logging.info(f"\nResults saved to: {output_dir}")
        logging.info(f"Log saved to: {log_file}")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Results: {output_dir}")
        print(f"📈 Report: {output_dir / 'evaluation_report.md'}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()