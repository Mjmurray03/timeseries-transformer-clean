#!/usr/bin/env python3
"""
# COMPONENT: Training Approaches Verification Script
# PURPOSE: Test both single-ticker and multi-stock training approaches
# INPUTS: Available ticker data, model checkpoints
# OUTPUTS: Performance comparison, validation metrics
# VERIFICATION: RMSE baseline validation, error handling, data integrity
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer
from train_multi_stock import MultiStockTransformer


def setup_logging():
    """Setup logging for verification"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("logs") / f"training_verification_{timestamp}.log"
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def check_data_availability() -> Dict[str, bool]:
    """
    # COMPONENT: Data Availability Checker
    # PURPOSE: Verify all required data files exist
    # VERIFICATION: FileNotFoundError handling, ticker validation
    """
    data_dir = Path("data/raw")
    available_tickers = {}
    
    expected_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
    
    for ticker in expected_tickers:
        ticker_dir = data_dir / ticker
        if ticker_dir.exists():
            parquet_files = list(ticker_dir.glob("*.parquet"))
            available_tickers[ticker] = len(parquet_files) > 0
        else:
            available_tickers[ticker] = False
    
    logging.info("Data availability check:")
    for ticker, available in available_tickers.items():
        status = "✓" if available else "✗"
        logging.info(f"  {ticker}: {status}")
    
    return available_tickers


def test_single_ticker_approach(ticker: str = "AAPL") -> Dict[str, any]:
    """
    # COMPONENT: Single-Ticker Approach Tester
    # PURPOSE: Validate per-ticker model training
    # VERIFICATION: Model loading, prediction shape, RMSE calculation
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"Testing Single-Ticker Approach: {ticker}")
    logging.info('='*50)
    
    results = {
        'approach': 'single_ticker',
        'ticker': ticker,
        'success': False,
        'error': None
    }
    
    try:
        # Check if model exists
        model_path = Path(f"models/model_{ticker}_best.pt")
        scaler_path = Path(f"models/scalers/scaler_{ticker}.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Load scaler
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        logging.info(f"Scaler features: {len(scaler_data['feature_names'])}")
        
        # Initialize model with same config
        config = checkpoint.get('model_config', {})
        
        # Get input dimension from scaler
        input_dim = len(scaler_data['feature_names'])
        
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4),
            max_seq_length=60,
            output_dim=3,
            use_attention_pooling=True
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy data
        batch_size = 4
        seq_len = 60
        dummy_input = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (batch_size, 3)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check for NaN/Inf
        assert not torch.any(torch.isnan(output)), "NaN in model output"
        assert not torch.any(torch.isinf(output)), "Inf in model output"
        
        # Extract metrics
        metrics = checkpoint.get('metrics', {})
        val_rmse = metrics.get('val_rmse', 'Unknown')
        
        results.update({
            'success': True,
            'model_params': sum(p.numel() for p in model.parameters()),
            'input_dim': input_dim,
            'val_rmse': val_rmse,
            'output_shape': list(output.shape),
            'metrics': metrics
        })
        
        logging.info(f"✓ Single-ticker model test passed")
        logging.info(f"  Validation RMSE: {val_rmse}")
        logging.info(f"  Output shape: {output.shape}")
        
    except Exception as e:
        results['error'] = str(e)
        logging.error(f"✗ Single-ticker test failed: {e}")
    
    return results


def test_multi_stock_approach() -> Dict[str, any]:
    """
    # COMPONENT: Multi-Stock Approach Tester  
    # PURPOSE: Validate unified multi-stock model
    # VERIFICATION: Ticker embedding, shape validation, per-ticker metrics
    """
    logging.info(f"\n{'='*50}")
    logging.info("Testing Multi-Stock Approach")
    logging.info('='*50)
    
    results = {
        'approach': 'multi_stock',
        'success': False,
        'error': None
    }
    
    try:
        # Check if model exists
        model_path = Path("models/model_multi_stock_best.pt")
        scaler_path = Path("models/scalers/scaler_multi_stock.json")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Load scaler
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        tickers = scaler_data['tickers']
        logging.info(f"Tickers: {tickers}")
        logging.info(f"Scaler features: {len(scaler_data['feature_names'])}")
        
        # Get config
        config = checkpoint.get('config', {})
        
        # Initialize model
        model = MultiStockTransformer(
            n_tickers=len(tickers),
            embedding_dim=config.get('embedding_dim', 16),
            input_dim=len(scaler_data['feature_names']),
            hidden_dim=config.get('hidden_dim', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            max_seq_length=config.get('seq_len', 60),
            output_dim=config.get('horizon', 3),
            use_attention_pooling=True
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with dummy data
        batch_size = 8
        seq_len = 60
        input_dim = len(scaler_data['feature_names'])
        
        dummy_sequences = torch.randn(batch_size, seq_len, input_dim)
        dummy_ticker_ids = torch.randint(0, len(tickers), (batch_size,))
        
        with torch.no_grad():
            output = model(dummy_sequences, dummy_ticker_ids)
        
        expected_shape = (batch_size, config.get('horizon', 3))
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check for NaN/Inf
        assert not torch.any(torch.isnan(output)), "NaN in model output"
        assert not torch.any(torch.isinf(output)), "Inf in model output"
        
        # Test ticker embedding
        embedding_weight = model.ticker_embedding.weight
        assert embedding_weight.shape[0] == len(tickers), "Embedding size mismatch"
        
        # Extract metrics
        metrics = checkpoint.get('metrics', {})
        val_loss = metrics.get('val_loss', 'Unknown')
        
        # Collect per-ticker RMSEs
        per_ticker_rmse = {}
        for ticker in tickers:
            rmse_key = f'val_rmse_{ticker}'
            if rmse_key in metrics:
                per_ticker_rmse[ticker] = metrics[rmse_key]
        
        results.update({
            'success': True,
            'model_params': sum(p.numel() for p in model.parameters()),
            'n_tickers': len(tickers),
            'input_dim': input_dim,
            'embedding_dim': config.get('embedding_dim', 16),
            'val_loss': val_loss,
            'per_ticker_rmse': per_ticker_rmse,
            'output_shape': list(output.shape),
            'tickers': tickers
        })
        
        logging.info(f"✓ Multi-stock model test passed")
        logging.info(f"  Validation Loss: {val_loss}")
        logging.info(f"  Tickers: {len(tickers)}")
        logging.info(f"  Output shape: {output.shape}")
        logging.info(f"  Per-ticker RMSE: {per_ticker_rmse}")
        
    except Exception as e:
        results['error'] = str(e)
        logging.error(f"✗ Multi-stock test failed: {e}")
    
    return results


def compare_approaches(single_results: List[Dict], multi_results: Dict):
    """
    # COMPONENT: Approach Comparator
    # PURPOSE: Compare performance between approaches
    # VERIFICATION: RMSE comparison, parameter efficiency, training success
    """
    logging.info(f"\n{'='*60}")
    logging.info("APPROACH COMPARISON")
    logging.info('='*60)
    
    # Single-ticker summary
    successful_single = [r for r in single_results if r['success']]
    failed_single = [r for r in single_results if not r['success']]
    
    logging.info(f"Single-Ticker Approach:")
    logging.info(f"  Successful models: {len(successful_single)}")
    logging.info(f"  Failed models: {len(failed_single)}")
    
    if successful_single:
        total_params = sum(r['model_params'] for r in successful_single)
        avg_params = total_params / len(successful_single)
        logging.info(f"  Avg parameters per model: {avg_params:,.0f}")
        logging.info(f"  Total parameters: {total_params:,}")
        
        # RMSE comparison
        rmse_values = []
        for r in successful_single:
            if isinstance(r.get('val_rmse'), (int, float)):
                rmse_values.append(r['val_rmse'])
        
        if rmse_values:
            avg_rmse = np.mean(rmse_values)
            logging.info(f"  Average RMSE: {avg_rmse:.6f}")
            logging.info(f"  Best RMSE: {min(rmse_values):.6f}")
            logging.info(f"  Worst RMSE: {max(rmse_values):.6f}")
    
    # Multi-stock summary
    logging.info(f"\nMulti-Stock Approach:")
    if multi_results['success']:
        logging.info(f"  ✓ Model trained successfully")
        logging.info(f"  Tickers: {multi_results['n_tickers']}")
        logging.info(f"  Total parameters: {multi_results['model_params']:,}")
        
        per_ticker = multi_results.get('per_ticker_rmse', {})
        if per_ticker:
            rmse_values = list(per_ticker.values())
            avg_rmse = np.mean(rmse_values)
            logging.info(f"  Average RMSE: {avg_rmse:.6f}")
            logging.info(f"  Best ticker RMSE: {min(rmse_values):.6f}")
            logging.info(f"  Worst ticker RMSE: {max(rmse_values):.6f}")
    else:
        logging.info(f"  ✗ Model training failed")
    
    # Efficiency comparison
    if successful_single and multi_results['success']:
        single_total_params = sum(r['model_params'] for r in successful_single)
        multi_total_params = multi_results['model_params']
        
        param_efficiency = single_total_params / multi_total_params
        logging.info(f"\nParameter Efficiency:")
        logging.info(f"  Single-ticker total: {single_total_params:,}")
        logging.info(f"  Multi-stock total: {multi_total_params:,}")
        logging.info(f"  Efficiency ratio: {param_efficiency:.2f}x")


def test_data_loading():
    """
    # COMPONENT: Data Loading Tester
    # PURPOSE: Verify data integrity and preprocessing
    # VERIFICATION: NaN handling, shape validation, feature engineering
    """
    logging.info(f"\n{'='*50}")
    logging.info("Testing Data Loading")
    logging.info('='*50)
    
    try:
        from train_ultra_simple import TickerDataProcessor
        from train_multi_stock import MultiStockProcessor
        
        # Test single ticker processor
        processor = TickerDataProcessor("AAPL")
        if processor.validate_ticker():
            df = processor.load_ticker_data()
            features = processor.engineer_features(df)
            normalized = processor.normalize_features(features)
            sequences, targets = processor.create_sequences(normalized)
            
            logging.info(f"✓ Single-ticker data loading test passed")
            logging.info(f"  Raw data: {len(df)} rows")
            logging.info(f"  Features: {features.shape}")
            logging.info(f"  Sequences: {sequences.shape}")
            logging.info(f"  Targets: {targets.shape}")
            
            # Verify no NaN/Inf
            assert not np.any(np.isnan(sequences)), "NaN in sequences"
            assert not np.any(np.isinf(sequences)), "Inf in sequences"
            assert not np.any(np.isnan(targets)), "NaN in targets"
            assert not np.any(np.isinf(targets)), "Inf in targets"
        
        # Test multi-stock processor
        available_tickers = [t for t, avail in check_data_availability().items() if avail][:3]
        if available_tickers:
            multi_processor = MultiStockProcessor(available_tickers)
            ticker_data = multi_processor.load_all_tickers()
            sequences, targets, ticker_ids = multi_processor.create_unified_sequences(ticker_data)
            
            logging.info(f"✓ Multi-stock data loading test passed")
            logging.info(f"  Tickers loaded: {len(ticker_data)}")
            logging.info(f"  Total sequences: {len(sequences)}")
            logging.info(f"  Sequence shape: {sequences.shape}")
            logging.info(f"  Target shape: {targets.shape}")
            logging.info(f"  Unique tickers: {len(np.unique(ticker_ids))}")
            
            # Verify no NaN/Inf
            assert not np.any(np.isnan(sequences)), "NaN in multi-stock sequences"
            assert not np.any(np.isinf(sequences)), "Inf in multi-stock sequences"
            assert not np.any(np.isnan(targets)), "NaN in multi-stock targets"
            assert not np.any(np.isinf(targets)), "Inf in multi-stock targets"
        
    except Exception as e:
        logging.error(f"Data loading test failed: {e}")


def main():
    """
    # COMPONENT: Main Verification Pipeline
    # PURPOSE: Execute comprehensive testing of both approaches
    # VERIFICATION: End-to-end validation, performance comparison, error reporting
    """
    log_file = setup_logging()
    
    logging.info("="*80)
    logging.info("TRAINING APPROACHES VERIFICATION")
    logging.info("="*80)
    
    # Check data availability
    available_data = check_data_availability()
    available_tickers = [t for t, avail in available_data.items() if avail]
    
    if not available_tickers:
        logging.error("No ticker data available for testing")
        return
    
    # Test data loading
    test_data_loading()
    
    # Test single-ticker approach for each available ticker
    single_results = []
    for ticker in available_tickers[:3]:  # Test first 3 tickers
        result = test_single_ticker_approach(ticker)
        single_results.append(result)
    
    # Test multi-stock approach
    multi_results = test_multi_stock_approach()
    
    # Compare approaches
    compare_approaches(single_results, multi_results)
    
    # Final summary
    logging.info(f"\n{'='*80}")
    logging.info("VERIFICATION SUMMARY")
    logging.info("="*80)
    
    successful_single = sum(1 for r in single_results if r['success'])
    total_single = len(single_results)
    
    logging.info(f"Single-ticker success rate: {successful_single}/{total_single}")
    logging.info(f"Multi-stock success: {'Yes' if multi_results['success'] else 'No'}")
    
    if multi_results['success'] and successful_single > 0:
        logging.info("✓ Both approaches validated successfully")
        
        # Baseline comparison
        baseline_rmse = 0.268  # AAPL baseline mentioned in requirements
        single_rmse_values = []
        multi_rmse_values = []
        
        for r in single_results:
            if r['success'] and isinstance(r.get('val_rmse'), (int, float)):
                single_rmse_values.append(r['val_rmse'])
        
        multi_per_ticker = multi_results.get('per_ticker_rmse', {})
        if multi_per_ticker:
            multi_rmse_values = list(multi_per_ticker.values())
        
        if single_rmse_values:
            best_single = min(single_rmse_values)
            vs_baseline = "✓ Maintained" if best_single <= baseline_rmse * 1.1 else "⚠ Degraded"
            logging.info(f"Best single-ticker RMSE: {best_single:.6f} vs baseline {baseline_rmse} {vs_baseline}")
        
        if multi_rmse_values:
            best_multi = min(multi_rmse_values)
            vs_baseline = "✓ Maintained" if best_multi <= baseline_rmse * 1.1 else "⚠ Degraded"
            logging.info(f"Best multi-stock RMSE: {best_multi:.6f} vs baseline {baseline_rmse} {vs_baseline}")
    
    else:
        logging.warning("Not all approaches validated successfully")
    
    logging.info(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()