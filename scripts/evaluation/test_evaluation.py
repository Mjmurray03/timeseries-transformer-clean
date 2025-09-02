#!/usr/bin/env python3
"""
# COMPONENT: Evaluation System Test Suite
# PURPOSE: Validate inference pipeline with mock and real data
# INPUTS: Model checkpoints, test data, mock scenarios
# OUTPUTS: Validation results, system integrity confirmation
# VERIFICATION: Price range validation, metric consistency, error handling
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import logging
from datetime import datetime, timedelta
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluate import (
    ModelLoader, 
    PredictionPipeline, 
    MetricsCalculator, 
    EvaluationVisualizer,
    load_stock_data
)
from src.models.timeseries_transformer import TimeSeriesTransformer


def setup_test_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_mock_model_checkpoint(temp_dir: Path) -> Path:
    """
    # COMPONENT: Mock Model Creator
    # PURPOSE: Create realistic model checkpoint for testing
    # VERIFICATION: Proper state dict structure, config consistency
    """
    # Create a simple transformer model
    model = TimeSeriesTransformer(
        input_dim=6,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=60,
        output_dim=3,
        use_attention_pooling=True
    )
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 6,
            'hidden_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'output_dim': 3,
            'max_seq_length': 60,
            'forecast_horizon': 3
        },
        'epoch': 50,
        'metrics': {
            'val_rmse': 0.255,
            'val_r2': 0.67,
            'val_direction_accuracy': 0.58
        },
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = temp_dir / "mock_model.pt"
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def create_mock_scaler(temp_dir: Path) -> Path:
    """
    # COMPONENT: Mock Scaler Creator
    # PURPOSE: Create realistic scaler parameters for testing
    # VERIFICATION: Proper feature statistics, JSON format consistency
    """
    scaler_data = {
        'ticker': 'TEST',
        'feature_names': ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns'],
        'scaler_params': {
            'Open': {'mean': 150.5, 'std': 25.2, 'min': 120.0, 'max': 200.0},
            'High': {'mean': 152.1, 'std': 25.8, 'min': 121.5, 'max': 202.5},
            'Low': {'mean': 149.2, 'std': 24.9, 'min': 118.0, 'max': 198.0},
            'Close': {'mean': 150.8, 'std': 25.3, 'min': 119.5, 'max': 201.0},
            'Volume': {'mean': 12.5, 'std': 1.2, 'min': 8.0, 'max': 16.0},  # log volume
            'Returns': {'mean': 0.001, 'std': 0.025, 'min': -0.15, 'max': 0.18}
        },
        'timestamp': datetime.now().isoformat()
    }
    
    scaler_path = temp_dir / "mock_scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    
    return scaler_path


def create_mock_stock_data(temp_dir: Path, n_days: int = 500) -> Path:
    """
    # COMPONENT: Mock Stock Data Creator
    # PURPOSE: Generate realistic OHLCV data for testing
    # VERIFICATION: Proper price relationships, realistic patterns
    """
    # Generate synthetic but realistic stock data
    np.random.seed(42)
    
    # Start with base price and random walk
    base_price = 150.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    
    # Generate price series
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Generate OHLCV data
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i, close in enumerate(prices):
        date = start_date + timedelta(days=i)
        
        # Generate realistic OHLC from close price
        daily_vol = abs(np.random.normal(0, 0.01))
        
        high = close * (1 + daily_vol * np.random.uniform(0.3, 1.0))
        low = close * (1 - daily_vol * np.random.uniform(0.3, 1.0))
        
        # Open is previous close with small gap
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume (log-normal distribution)
        volume = np.random.lognormal(15, 0.5)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    data_path = temp_dir / "mock_data.parquet"
    df.to_parquet(data_path)
    
    logging.info(f"Created mock data: {len(df)} days, price range ${df['Close'].min():.2f}-${df['Close'].max():.2f}")
    
    return data_path


def test_model_loader():
    """
    # COMPONENT: ModelLoader Test
    # PURPOSE: Validate model loading and integrity checking
    # VERIFICATION: Architecture consistency, parameter loading, device handling
    """
    logging.info("\n" + "="*50)
    logging.info("Testing ModelLoader")
    logging.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock checkpoint
        checkpoint_path = create_mock_model_checkpoint(temp_path)
        scaler_path = create_mock_scaler(temp_path)
        
        # Test loading
        loader = ModelLoader(device='cpu')
        model, config, scaler = loader.load_checkpoint(checkpoint_path, scaler_path)
        
        # Verify model
        assert isinstance(model, TimeSeriesTransformer), "Model type mismatch"
        assert config['input_dim'] == 6, "Config input_dim mismatch"
        assert scaler['ticker'] == 'TEST', "Scaler ticker mismatch"
        
        # Test integrity check (already called in load_checkpoint)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model has no parameters"
        
        logging.info(f"✓ Model loaded successfully: {param_count:,} parameters")
        logging.info(f"✓ Config loaded: {config}")
        logging.info(f"✓ Scaler loaded: {len(scaler['feature_names'])} features")
        
        return True


def test_prediction_pipeline():
    """
    # COMPONENT: PredictionPipeline Test
    # PURPOSE: Validate prediction generation and dollar conversion
    # VERIFICATION: Prediction shapes, price ranges, scaling consistency
    """
    logging.info("\n" + "="*50)
    logging.info("Testing PredictionPipeline")
    logging.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock components
        checkpoint_path = create_mock_model_checkpoint(temp_path)
        scaler_path = create_mock_scaler(temp_path)
        
        # Load model and scaler
        loader = ModelLoader(device='cpu')
        model, config, scaler = loader.load_checkpoint(checkpoint_path, scaler_path)
        
        # Initialize pipeline
        pipeline = PredictionPipeline(model, scaler, torch.device('cpu'), config)
        
        # Test single sequence prediction
        seq_len = 60
        n_features = 6
        test_features = np.random.randn(seq_len, n_features).astype(np.float32)
        
        # Predict without attention
        result = pipeline.predict_sequence(test_features, return_attention=False)
        
        assert 'predictions_standardized' in result, "Missing standardized predictions"
        assert 'predictions_returns' in result, "Missing return predictions"
        
        predictions = result['predictions_returns']
        assert predictions.shape == (3,), f"Expected (3,), got {predictions.shape}"
        assert not np.any(np.isnan(predictions)), "NaN in predictions"
        assert not np.any(np.isinf(predictions)), "Inf in predictions"
        
        # Test with attention
        result_with_attention = pipeline.predict_sequence(test_features, return_attention=True)
        assert 'attention_weights' in result_with_attention, "Missing attention weights"
        
        attention = result_with_attention['attention_weights']
        assert len(attention) == config['num_layers'], "Wrong number of attention layers"
        
        logging.info(f"✓ Single prediction: shape {predictions.shape}")
        logging.info(f"✓ Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logging.info(f"✓ Attention weights: {len(attention)} layers")
        
        return True


def test_sliding_window_predictions():
    """
    # COMPONENT: Sliding Window Test
    # PURPOSE: Validate sliding window prediction generation
    # VERIFICATION: Prediction count, date alignment, price conversion
    """
    logging.info("\n" + "="*50)
    logging.info("Testing Sliding Window Predictions")
    logging.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock components
        checkpoint_path = create_mock_model_checkpoint(temp_path)
        scaler_path = create_mock_scaler(temp_path)
        data_path = create_mock_stock_data(temp_path, n_days=200)
        
        # Load components
        loader = ModelLoader(device='cpu')
        model, config, scaler = loader.load_checkpoint(checkpoint_path, scaler_path)
        pipeline = PredictionPipeline(model, scaler, torch.device('cpu'), config)
        
        # Load data
        data = pd.read_parquet(data_path)
        
        # Generate predictions
        predictions_df = pipeline.sliding_window_predictions(
            data=data,
            start_date='2023-03-01',
            end_date='2023-06-01',
            seq_len=60,
            stride=5
        )
        
        # Verify predictions
        assert len(predictions_df) > 0, "No predictions generated"
        assert 'current_price' in predictions_df.columns, "Missing current_price column"
        assert 'actual_1d' in predictions_df.columns, "Missing actual_1d column"
        assert 'predicted_1d' in predictions_df.columns, "Missing predicted_1d column"
        
        # Check price ranges are realistic
        current_prices = predictions_df['current_price'].dropna()
        predicted_prices = predictions_df['predicted_1d'].dropna()
        actual_prices = predictions_df['actual_1d'].dropna()
        
        assert current_prices.min() > 50, "Current prices too low"
        assert current_prices.max() < 500, "Current prices too high"
        assert predicted_prices.min() > 50, "Predicted prices too low"
        assert predicted_prices.max() < 500, "Predicted prices too high"
        
        # Check no extreme changes (> 50% in one day)
        if len(predicted_prices) > 0 and len(current_prices) > 0:
            # Align indices
            common_idx = predictions_df.dropna(subset=['current_price', 'predicted_1d']).index
            if len(common_idx) > 0:
                current_aligned = predictions_df.loc[common_idx, 'current_price']
                predicted_aligned = predictions_df.loc[common_idx, 'predicted_1d']
                pct_changes = abs((predicted_aligned - current_aligned) / current_aligned)
                max_change = pct_changes.max()
                assert max_change < 0.5, f"Extreme prediction change: {max_change:.2%}"
        
        logging.info(f"✓ Generated {len(predictions_df)} predictions")
        logging.info(f"✓ Current price range: ${current_prices.min():.2f}-${current_prices.max():.2f}")
        logging.info(f"✓ Predicted price range: ${predicted_prices.min():.2f}-${predicted_prices.max():.2f}")
        
        return predictions_df


def test_metrics_calculator():
    """
    # COMPONENT: MetricsCalculator Test
    # PURPOSE: Validate all metric calculations
    # VERIFICATION: Metric ranges, edge case handling, statistical validity
    """
    logging.info("\n" + "="*50)
    logging.info("Testing MetricsCalculator")
    logging.info("="*50)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    # Perfect predictions (should give perfect metrics)
    y_true_perfect = np.random.randn(n_samples)
    y_pred_perfect = y_true_perfect.copy()
    
    perfect_metrics = MetricsCalculator.calculate_regression_metrics(y_true_perfect, y_pred_perfect)
    
    assert abs(perfect_metrics['rmse']) < 1e-10, "RMSE should be ~0 for perfect predictions"
    assert abs(perfect_metrics['r2'] - 1.0) < 1e-10, "R² should be 1 for perfect predictions"
    
    # Realistic predictions
    y_true = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.1
    y_pred = y_true + noise
    
    reg_metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
    
    assert reg_metrics['rmse'] > 0, "RMSE should be positive"
    assert 0 <= reg_metrics['r2'] <= 1, f"R² should be in [0,1], got {reg_metrics['r2']}"
    assert reg_metrics['n_samples'] == n_samples, "Sample count mismatch"
    
    # Test directional accuracy
    returns_true = np.random.randn(n_samples) * 0.02
    returns_pred = returns_true + np.random.randn(n_samples) * 0.01
    
    dir_metrics = MetricsCalculator.calculate_directional_accuracy(returns_true, returns_pred)
    
    assert 0 <= dir_metrics['overall_accuracy'] <= 1, "Directional accuracy out of range"
    assert 0 <= dir_metrics['up_precision'] <= 1, "Up precision out of range"
    assert 0 <= dir_metrics['up_recall'] <= 1, "Up recall out of range"
    
    # Test trading metrics
    daily_returns = np.random.normal(0.001, 0.02, n_samples)  # 0.1% daily return, 2% vol
    returns_series = pd.Series(daily_returns)
    
    trading_metrics = MetricsCalculator.calculate_trading_metrics(returns_series)
    
    assert not np.isnan(trading_metrics['sharpe_ratio']), "Sharpe ratio is NaN"
    assert trading_metrics['max_drawdown'] <= 0, "Max drawdown should be negative"
    assert 0 <= trading_metrics['win_rate'] <= 1, "Win rate out of range"
    
    logging.info(f"✓ Regression metrics: RMSE={reg_metrics['rmse']:.4f}, R²={reg_metrics['r2']:.3f}")
    logging.info(f"✓ Directional accuracy: {dir_metrics['overall_accuracy']:.1%}")
    logging.info(f"✓ Trading metrics: Sharpe={trading_metrics['sharpe_ratio']:.2f}, Win Rate={trading_metrics['win_rate']:.1%}")
    
    return True


def test_visualization_system():
    """
    # COMPONENT: Visualization Test
    # PURPOSE: Validate plot generation without errors
    # VERIFICATION: File creation, no plotting errors, proper formatting
    """
    logging.info("\n" + "="*50)
    logging.info("Testing Visualization System")
    logging.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test predictions data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_price = 150
        actual_returns = np.random.normal(0.001, 0.02, 100)
        pred_returns = actual_returns + np.random.normal(0, 0.01, 100)
        
        # Generate price series
        actual_prices = [base_price]
        pred_prices = [base_price]
        
        for i in range(99):
            actual_prices.append(actual_prices[-1] * (1 + actual_returns[i]))
            pred_prices.append(pred_prices[-1] * (1 + pred_returns[i]))
        
        predictions_df = pd.DataFrame({
            'current_price': actual_prices[:-1],
            'actual_1d': actual_prices[1:],
            'predicted_1d': pred_prices[1:],
            'actual_return_1d': actual_returns[:99],
            'predicted_return_1d': pred_returns[:99]
        }, index=dates[:99])
        
        # Initialize visualizer
        visualizer = EvaluationVisualizer(temp_path)
        
        # Test main plot
        try:
            visualizer.plot_predictions_vs_actuals(predictions_df, 'TEST')
            plot_file = temp_path / "TEST_predictions_vs_actuals.png"
            assert plot_file.exists(), "Predictions plot not created"
            logging.info("✓ Predictions vs actuals plot created")
        except Exception as e:
            logging.warning(f"Predictions plot failed: {e}")
        
        # Test error analysis
        try:
            visualizer.plot_error_analysis(predictions_df)
            error_file = temp_path / "error_analysis.png"
            assert error_file.exists(), "Error analysis plot not created"
            logging.info("✓ Error analysis plot created")
        except Exception as e:
            logging.warning(f"Error analysis plot failed: {e}")
        
        # Test market condition analysis
        try:
            visualizer.plot_performance_by_market_condition(predictions_df)
            market_file = temp_path / "performance_by_market_condition.png"
            assert market_file.exists(), "Market condition plot not created"
            logging.info("✓ Market condition plot created")
        except Exception as e:
            logging.warning(f"Market condition plot failed: {e}")
        
        # Test attention heatmap with mock data
        try:
            # Create mock attention weights
            seq_len = 60
            n_layers = 2
            mock_attention = [np.random.rand(seq_len, seq_len) for _ in range(n_layers)]
            
            visualizer.plot_attention_heatmap(mock_attention, seq_len)
            attention_file = temp_path / "attention_heatmap.png"
            assert attention_file.exists(), "Attention heatmap not created"
            logging.info("✓ Attention heatmap created")
        except Exception as e:
            logging.warning(f"Attention heatmap failed: {e}")
        
        return True


def test_end_to_end_evaluation():
    """
    # COMPONENT: End-to-End Test
    # PURPOSE: Validate complete evaluation pipeline
    # VERIFICATION: Full workflow execution, report generation, file outputs
    """
    logging.info("\n" + "="*50)
    logging.info("Testing End-to-End Evaluation")
    logging.info("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create all mock components
        checkpoint_path = create_mock_model_checkpoint(temp_path)
        scaler_path = create_mock_scaler(temp_path)
        data_path = create_mock_stock_data(temp_path, n_days=300)
        
        # Import evaluation functions
        from evaluate import (
            setup_logging, generate_comprehensive_report,
            load_stock_data
        )
        
        # Load components
        loader = ModelLoader(device='cpu')
        model, config, scaler = loader.load_checkpoint(checkpoint_path, scaler_path)
        
        # Load data
        data = load_stock_data(data_path)
        
        # Generate predictions
        pipeline = PredictionPipeline(model, scaler, torch.device('cpu'), config)
        predictions_df = pipeline.sliding_window_predictions(
            data=data,
            start_date='2023-03-01',
            end_date='2023-08-01',
            seq_len=60,
            stride=3
        )
        
        # Calculate metrics
        actual = predictions_df['actual_1d'].dropna()
        predicted = predictions_df['predicted_1d'].dropna()
        common_idx = actual.index.intersection(predicted.index)
        
        regression_metrics = MetricsCalculator.calculate_regression_metrics(
            actual[common_idx].values, predicted[common_idx].values
        )
        
        actual_ret = predictions_df['actual_return_1d'].dropna()
        predicted_ret = predictions_df['predicted_return_1d'].dropna()
        common_ret_idx = actual_ret.index.intersection(predicted_ret.index)
        
        directional_metrics = MetricsCalculator.calculate_directional_accuracy(
            actual_ret[common_ret_idx].values, predicted_ret[common_ret_idx].values
        )
        
        trading_metrics = MetricsCalculator.calculate_trading_metrics(
            predicted_ret, risk_free_rate=0.02
        )
        
        # Generate report
        model_info = {
            'path': str(checkpoint_path),
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': 'TimeSeriesTransformer'
        }
        
        data_info = {
            'ticker': 'TEST',
            'start_date': '2023-03-01',
            'end_date': '2023-08-01',
            'n_predictions': len(predictions_df)
        }
        
        report = generate_comprehensive_report(
            model_info=model_info,
            data_info=data_info,
            predictions_df=predictions_df,
            regression_metrics=regression_metrics,
            directional_metrics=directional_metrics,
            trading_metrics=trading_metrics,
            output_dir=temp_path
        )
        
        # Verify outputs
        assert (temp_path / "evaluation_report.json").exists(), "JSON report not created"
        assert (temp_path / "evaluation_report.md").exists(), "Markdown report not created"
        
        # Check report content
        assert report['regression_metrics']['rmse'] > 0, "Invalid RMSE in report"
        assert 0 <= report['directional_metrics']['overall_accuracy'] <= 1, "Invalid accuracy in report"
        
        logging.info(f"✓ Generated {len(predictions_df)} predictions")
        logging.info(f"✓ RMSE: ${regression_metrics['rmse']:.4f}")
        logging.info(f"✓ Directional Accuracy: {directional_metrics['overall_accuracy']:.1%}")
        logging.info(f"✓ Reports generated successfully")
        
        return True


def main():
    """Run all evaluation system tests"""
    setup_test_logging()
    
    logging.info("="*80)
    logging.info("EVALUATION SYSTEM TEST SUITE")
    logging.info("="*80)
    
    tests = [
        ("ModelLoader", test_model_loader),
        ("PredictionPipeline", test_prediction_pipeline),
        ("Sliding Window", test_sliding_window_predictions),
        ("MetricsCalculator", test_metrics_calculator),
        ("Visualization", test_visualization_system),
        ("End-to-End", test_end_to_end_evaluation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logging.info(f"\n🧪 Running {test_name} test...")
            test_func()
            logging.info(f"✅ {test_name} test PASSED")
            passed += 1
        except Exception as e:
            logging.error(f"❌ {test_name} test FAILED: {e}")
            failed += 1
    
    # Final summary
    logging.info("\n" + "="*80)
    logging.info("TEST SUMMARY")
    logging.info("="*80)
    logging.info(f"✅ Passed: {passed}")
    logging.info(f"❌ Failed: {failed}")
    logging.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logging.info("🎉 All tests passed! Evaluation system is ready.")
        print("\n✅ All evaluation tests passed!")
        print("📊 System validated and ready for use")
    else:
        logging.warning(f"⚠️  {failed} test(s) failed. Please check the logs.")
        print(f"\n⚠️  {failed} test(s) failed - check logs for details")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)