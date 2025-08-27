"""
Test BacktestEngine core functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.backtesting.portfolio import Portfolio
from src.backtesting.market_simulator import MarketSimulator
from src.backtesting.strategy import Strategy
from src.backtesting.risk_manager import RiskManager


class TestBacktestConfig:
    """Test BacktestConfig functionality"""
    
    def test_config_initialization(self):
        """Test configuration initializes correctly"""
        config = BacktestConfig(
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_params={"test": "value"},
            risk_params={"max_risk": 0.2},
            market_params={"cost_model": {}}
        )
        
        assert config.initial_capital == 100000.0
        assert config.start_date == "2023-01-01"
        assert config.end_date == "2023-12-31"
        assert config.entry_rules is not None
        assert config.exit_rules is not None
        
    def test_config_default_rules(self):
        """Test default entry and exit rules are set"""
        config = BacktestConfig(
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_params={},
            risk_params={},
            market_params={}
        )
        
        assert config.entry_rules["prediction_threshold"] == 0.02
        assert config.exit_rules["stop_loss"] == 0.02


class TestBacktestEngine:
    """Test BacktestEngine functionality"""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample backtest configuration"""
        return BacktestConfig(
            initial_capital=100000.0,
            start_date="2023-01-01",
            end_date="2023-01-31",
            strategy_params={
                "min_expected_return": 0.02,
                "min_confidence": 0.7,
                "max_positions": 5
            },
            risk_params={
                "max_portfolio_risk": 0.2,
                "max_position_size": 0.1
            },
            market_params={
                "cost_model": {
                    "commission": {"fixed": 1.0, "percentage": 0.001},
                    "slippage": {"base": 0.0005}
                }
            }
        )
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction data"""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq='D')
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        data = {}
        for date in dates:
            data[date] = {}
            for ticker in tickers:
                data[date][ticker] = {
                    'return_5d': np.random.normal(0.01, 0.05),
                    'confidence': np.random.uniform(0.6, 0.9),
                    'volatility': np.random.uniform(0.15, 0.35)
                }
        
        # Convert to DataFrame
        df_data = []
        for date, tickers_data in data.items():
            for ticker, pred_data in tickers_data.items():
                row = {'date': date, 'ticker': ticker}
                row.update(pred_data)
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df.pivot(index='date', columns='ticker', values=['return_5d', 'confidence', 'volatility'])
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq='D')
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        data = []
        for date in dates:
            for ticker in tickers:
                base_price = {"AAPL": 150, "MSFT": 250, "GOOGL": 100}[ticker]
                price = base_price * (1 + np.random.normal(0, 0.02))
                
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'open': price * 0.995,
                    'high': price * 1.005,
                    'low': price * 0.99,
                    'close': price,
                    'volume': np.random.randint(1000000, 5000000),
                    'volatility': np.random.uniform(0.15, 0.35)
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(['date', 'ticker'])
        return df
    
    def test_engine_initialization(self, sample_config):
        """Test engine initializes correctly"""
        engine = BacktestEngine(sample_config)
        
        assert engine.config == sample_config
        assert isinstance(engine.strategy, Strategy)
        assert isinstance(engine.portfolio, Portfolio)
        assert isinstance(engine.market_sim, MarketSimulator)
        assert isinstance(engine.risk_manager, RiskManager)
        assert engine.results == []
        assert engine.trades_log == []
    
    def test_run_backtest_basic(self, sample_config, sample_predictions, sample_market_data):
        """Test basic backtest execution"""
        engine = BacktestEngine(sample_config)
        
        # Mock the components to avoid complex interactions
        engine.strategy.generate_signals = Mock(return_value=[])
        engine.risk_manager.filter_signals = Mock(return_value=[])
        engine.market_sim.execute_orders = Mock(return_value=[])
        
        # Flatten predictions for simpler format
        predictions_flat = sample_predictions['return_5d'].fillna(0)
        
        results = engine.run(predictions_flat, sample_market_data)
        
        assert isinstance(results, dict)
        assert 'config' in results
        assert 'metrics' in results
        assert 'portfolio_history' in results
    
    def test_walk_forward_analysis(self, sample_config):
        """Test walk-forward analysis functionality"""
        engine = BacktestEngine(sample_config)
        
        # Create larger dataset for walk-forward
        dates = pd.date_range("2023-01-01", "2023-12-31", freq='D')
        predictions = pd.DataFrame(
            np.random.randn(len(dates), 3) * 0.02,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"]
        )
        
        market_data = pd.DataFrame({
            ('AAPL', 'close'): np.random.randn(len(dates)) * 10 + 150,
            ('MSFT', 'close'): np.random.randn(len(dates)) * 15 + 250,
            ('GOOGL', 'close'): np.random.randn(len(dates)) * 8 + 100
        }, index=dates)
        
        # Mock the run method to avoid full execution
        engine.run = Mock(return_value={
            'metrics': {
                'total_return': 0.05,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08
            }
        })
        
        results = engine.run_walk_forward_analysis(
            predictions, market_data,
            train_window=60, test_window=20, step_size=10
        )
        
        assert 'aggregate_metrics' in results
        assert 'period_results' in results
        assert 'walk_forward_config' in results
        
    def test_generate_report(self, sample_config):
        """Test report generation"""
        engine = BacktestEngine(sample_config)
        
        # Create mock results
        sample_results = [
            {
                'date': datetime(2023, 1, 1),
                'portfolio_value': 100000,
                'trades': []
            },
            {
                'date': datetime(2023, 1, 2),
                'portfolio_value': 101000,
                'trades': []
            }
        ]
        
        report = engine.generate_report(sample_results)
        
        assert isinstance(report, dict)
        assert 'config' in report
        assert 'metrics' in report
        assert 'portfolio_history' in report
        
        # Check metrics calculations
        metrics = report['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine"""
    
    def test_end_to_end_backtest(self):
        """Test complete end-to-end backtest"""
        # Create simple test data
        config = BacktestConfig(
            initial_capital=10000.0,
            start_date="2023-01-01",
            end_date="2023-01-05",
            strategy_params={"min_expected_return": 0.01, "min_confidence": 0.5},
            risk_params={"max_portfolio_risk": 0.5},
            market_params={"cost_model": {}}
        )
        
        # Simple predictions: always predict positive returns
        dates = pd.date_range("2023-01-01", "2023-01-05", freq='D')
        predictions = pd.DataFrame({
            'AAPL': [0.02] * len(dates)
        }, index=dates)
        
        # Simple market data
        market_data_list = []
        for i, date in enumerate(dates):
            market_data_list.append({
                'date': date,
                'ticker': 'AAPL',
                'close': 100 + i,
                'volume': 1000000,
                'volatility': 0.2
            })
        
        market_data = pd.DataFrame(market_data_list).set_index(['date', 'ticker'])
        
        engine = BacktestEngine(config)
        results = engine.run(predictions, market_data)
        
        # Basic validation
        assert results['metrics']['final_value'] >= 0
        assert len(results['portfolio_history']) == len(dates)
        
    def test_error_handling(self, sample_config):
        """Test error handling in backtest execution"""
        engine = BacktestEngine(sample_config)
        
        # Test with empty data
        empty_predictions = pd.DataFrame()
        empty_market_data = pd.DataFrame()
        
        results = engine.run(empty_predictions, empty_market_data)
        
        # Should handle gracefully
        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__])