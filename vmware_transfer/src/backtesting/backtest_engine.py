"""
BacktestEngine - Main orchestrator for backtesting framework
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .portfolio import Portfolio
from .market_simulator import MarketSimulator
from .strategy import Strategy
from .risk_manager import RiskManager
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float
    start_date: str
    end_date: str
    strategy_params: Dict[str, Any]
    risk_params: Dict[str, Any]
    market_params: Dict[str, Any]
    
    # Strategy configuration
    entry_rules: Dict[str, Any] = None
    exit_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.entry_rules is None:
            self.entry_rules = {
                "prediction_threshold": 0.02,  # 2% expected return
                "confidence_threshold": 0.7,   # 70% confidence
                "max_positions": 10,           # Portfolio limit
                "position_size": 0.1           # 10% per position
            }
        
        if self.exit_rules is None:
            self.exit_rules = {
                "profit_target": 0.05,         # 5% profit
                "stop_loss": 0.02,             # 2% loss
                "time_stop": 5,                # Days
                "trailing_stop": True
            }


class BacktestEngine:
    """Main backtesting orchestrator"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy = Strategy(config.strategy_params)
        self.portfolio = Portfolio(config.initial_capital)
        self.market_sim = MarketSimulator(config.market_params)
        self.risk_manager = RiskManager(config.risk_params)
        self.metrics_tracker = MetricsTracker()
        
        self.results: List[Dict] = []
        self.trades_log: List[Dict] = []
        
        logger.info(f"BacktestEngine initialized with ${config.initial_capital:,.2f} initial capital")
        
    def run(self, predictions: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete backtest
        
        Args:
            predictions: DataFrame with model predictions indexed by date and tickers as columns
            market_data: DataFrame with OHLCV data indexed by date with MultiIndex (date, ticker)
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Filter data to backtest period
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        predictions = predictions.loc[start_date:end_date]
        market_data = market_data.loc[start_date:end_date]
        
        # Ensure data alignment
        common_dates = predictions.index.intersection(market_data.index.get_level_values(0).unique())
        
        logger.info(f"Processing {len(common_dates)} trading days")
        
        for date in common_dates:
            self._process_trading_day(date, predictions, market_data)
            
        # Calculate final metrics
        final_results = self.generate_report(self.results)
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio.total_value:,.2f}")
        logger.info(f"Total return: {final_results['metrics']['total_return']:.2%}")
        
        return final_results
    
    def _process_trading_day(self, date: pd.Timestamp, predictions: pd.DataFrame, market_data: pd.DataFrame):
        """Process a single trading day"""
        try:
            # Get predictions for date
            if date not in predictions.index:
                return
                
            daily_predictions = predictions.loc[date]
            
            # Get market data for date
            try:
                daily_market_data = market_data.loc[date]
            except KeyError:
                logger.warning(f"No market data for {date}")
                return
            
            # Generate signals
            signals = self.strategy.generate_signals(
                daily_predictions,
                self.portfolio.positions,
                market_data.loc[:date]
            )
            
            if signals:
                logger.debug(f"{date}: Generated {len(signals)} signals")
            
            # Risk checks
            approved_signals = self.risk_manager.filter_signals(
                signals,
                self.portfolio,
                daily_market_data
            )
            
            if len(approved_signals) < len(signals):
                logger.debug(f"{date}: Risk manager filtered {len(signals) - len(approved_signals)} signals")
            
            # Execute trades
            executed_trades = self.market_sim.execute_orders(
                approved_signals,
                daily_market_data,
                self.portfolio
            )
            
            # Log trades
            for trade in executed_trades:
                self.trades_log.append({
                    'date': date,
                    'ticker': trade.ticker,
                    'type': trade.type,
                    'shares': trade.shares,
                    'execution_price': trade.execution_price,
                    'commission': trade.commission,
                    'slippage': trade.slippage,
                    'total_cost': trade.total_cost
                })
            
            # Update portfolio
            self.portfolio.update(executed_trades, daily_market_data)
            
            # Track metrics
            self.metrics_tracker.update(self.portfolio, date)
            
            # Record daily state
            portfolio_snapshot = {
                'date': date,
                'portfolio_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_value': sum(p.current_value for p in self.portfolio.positions.values()),
                'num_positions': len(self.portfolio.positions),
                'daily_return': self.portfolio.daily_return if hasattr(self.portfolio, 'daily_return') else 0.0,
                'positions': {ticker: {
                    'shares': pos.shares,
                    'current_price': pos.current_price,
                    'current_value': pos.current_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                } for ticker, pos in self.portfolio.positions.items()},
                'trades': executed_trades
            }
            
            self.results.append(portfolio_snapshot)
            
        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
            raise
    
    def generate_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        if not results:
            return {"error": "No results to report"}
        
        # Extract portfolio values for analysis
        portfolio_values = [r['portfolio_value'] for r in results]
        dates = [r['date'] for r in results]
        
        # Calculate returns
        returns = pd.Series(portfolio_values, index=dates).pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Trading metrics
        all_trades = [trade for r in results for trade in r['trades']]
        trade_analysis = self._analyze_trades(all_trades)
        
        report = {
            'config': {
                'initial_capital': self.config.initial_capital,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'strategy_params': self.config.strategy_params
            },
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': portfolio_values[-1],
                'total_trades': len(all_trades),
                **trade_analysis
            },
            'portfolio_history': results,
            'trades_log': self.trades_log,
            'daily_returns': returns.to_dict(),
            'equity_curve': {str(date): value for date, value in zip(dates, portfolio_values)}
        }
        
        return report
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _analyze_trades(self, trades: List) -> Dict[str, Any]:
        """Analyze trade statistics"""
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'profit_factor': 0.0,
                'avg_winner': 0.0,
                'avg_loser': 0.0
            }
        
        # For now, return basic trade counts
        # TODO: Implement full trade P&L analysis when positions are closed
        return {
            'win_rate': 0.0,  # Will calculate after implementing position closing logic
            'avg_trade': 0.0,
            'profit_factor': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'buy_trades': len([t for t in trades if t.type == 'BUY']),
            'sell_trades': len([t for t in trades if t.type == 'SELL'])
        }
    
    def run_walk_forward_analysis(self, predictions: pd.DataFrame, market_data: pd.DataFrame,
                                  train_window: int = 252, test_window: int = 63, 
                                  step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward analysis as specified in design.md
        
        Args:
            predictions: Model predictions
            market_data: Market data
            train_window: Training window in days (252 = 1 year)
            test_window: Test window in days (63 = 3 months)
            step_size: Step size in days (21 = 1 month)
            
        Returns:
            Walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis with {train_window}d train, {test_window}d test windows")
        
        results = []
        start_idx = train_window
        
        while start_idx + test_window < len(predictions):
            # Define windows
            train_start = start_idx - train_window
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + test_window
            
            # Get date ranges
            train_dates = predictions.index[train_start:train_end]
            test_dates = predictions.index[test_start:test_end]
            
            logger.info(f"Walk-forward period: {test_dates[0]} to {test_dates[-1]}")
            
            # Run backtest on test period
            test_config = BacktestConfig(
                initial_capital=self.config.initial_capital,
                start_date=str(test_dates[0].date()),
                end_date=str(test_dates[-1].date()),
                strategy_params=self.config.strategy_params,
                risk_params=self.config.risk_params,
                market_params=self.config.market_params
            )
            
            # Create new engine for this period
            period_engine = BacktestEngine(test_config)
            period_result = period_engine.run(predictions, market_data)
            
            results.append({
                'period': f"{test_dates[0].date()}_{test_dates[-1].date()}",
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'metrics': period_result['metrics']
            })
            
            start_idx += step_size
        
        # Aggregate results
        all_returns = [r['metrics']['total_return'] for r in results]
        all_sharpe = [r['metrics']['sharpe_ratio'] for r in results if r['metrics']['sharpe_ratio'] != 0]
        
        aggregate_metrics = {
            'num_periods': len(results),
            'avg_return': np.mean(all_returns),
            'std_return': np.std(all_returns),
            'avg_sharpe': np.mean(all_sharpe) if all_sharpe else 0.0,
            'win_rate': len([r for r in all_returns if r > 0]) / len(all_returns),
            'best_period': max(all_returns),
            'worst_period': min(all_returns)
        }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'period_results': results,
            'walk_forward_config': {
                'train_window': train_window,
                'test_window': test_window,
                'step_size': step_size
            }
        }