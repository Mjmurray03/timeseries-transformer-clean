"""
Backtesting Framework for Time Series Transformer

This module provides a comprehensive backtesting framework for evaluating
trading strategies based on model predictions.
"""

from .backtest_engine import BacktestConfig, BacktestEngine
from .market_simulator import ExecutedTrade, MarketSimulator, OrderExecution
from .metrics import MetricsTracker, PerformanceAnalyzer
from .portfolio import Portfolio, Position
from .reporting import ReportGenerator
from .risk_manager import RiskManager
from .strategy import Strategy, TradingSignal

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "Portfolio",
    "Position",
    "MarketSimulator",
    "OrderExecution",
    "ExecutedTrade",
    "Strategy",
    "TradingSignal",
    "RiskManager",
    "MetricsTracker",
    "PerformanceAnalyzer",
    "ReportGenerator",
]
