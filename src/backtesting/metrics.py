"""
Performance metrics calculation and tracking
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of portfolio performance at a point in time"""

    date: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    volatility: float


class MetricsTracker:
    """Track performance metrics over time"""

    def __init__(self):
        self.snapshots: List[PerformanceSnapshot] = []
        self.portfolio_values: List[float] = []
        self.dates: List[datetime] = []
        self.daily_returns: List[float] = []
        self.running_max: List[float] = []

        logger.info("MetricsTracker initialized")

    def update(self, portfolio, date: datetime):
        """Update metrics with current portfolio state"""
        try:
            current_value = portfolio.total_value

            # Calculate daily return
            if self.portfolio_values:
                daily_return = (current_value - self.portfolio_values[-1]) / self.portfolio_values[
                    -1
                ]
            else:
                daily_return = 0.0

            # Update running maximum for drawdown calculation
            if self.running_max:
                running_max = max(self.running_max[-1], current_value)
            else:
                running_max = current_value

            # Calculate current drawdown
            drawdown = (running_max - current_value) / running_max if running_max > 0 else 0.0

            # Calculate cumulative return
            initial_value = portfolio.initial_capital
            cumulative_return = (current_value - initial_value) / initial_value

            # Calculate rolling volatility (last 30 days)
            if len(self.daily_returns) >= 30:
                volatility = np.std(self.daily_returns[-30:]) * np.sqrt(252)
            else:
                volatility = (
                    np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0.0
                )

            # Create snapshot
            snapshot = PerformanceSnapshot(
                date=date,
                portfolio_value=current_value,
                cash=portfolio.cash,
                positions_value=portfolio.positions_value,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                drawdown=drawdown,
                volatility=volatility,
            )

            # Store data
            self.snapshots.append(snapshot)
            self.portfolio_values.append(current_value)
            self.dates.append(date)
            self.daily_returns.append(daily_return)
            self.running_max.append(running_max)

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]

        return {
            "portfolio_value": latest.portfolio_value,
            "cumulative_return": latest.cumulative_return,
            "daily_return": latest.daily_return,
            "current_drawdown": latest.drawdown,
            "volatility": latest.volatility,
        }


class PerformanceAnalyzer:
    """
    Calculate comprehensive performance metrics following design.md specifications
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analyzer

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"PerformanceAnalyzer initialized with risk-free rate: {risk_free_rate:.2%}")

    def calculate_metrics(
        self, portfolio_history: List[Dict], closed_positions: List = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics as specified in design.md

        Args:
            portfolio_history: List of portfolio snapshots over time
            closed_positions: List of closed positions for trade analysis

        Returns:
            Dictionary with all performance metrics
        """
        if not portfolio_history:
            logger.warning("No portfolio history provided")
            return {}

        # Extract time series data
        values = [h["portfolio_value"] for h in portfolio_history]
        dates = [h["date"] for h in portfolio_history]

        # Calculate returns
        returns = pd.Series(values, index=dates).pct_change().dropna()

        # Core metrics
        core_metrics = self._calculate_core_metrics(values, returns)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(values, returns)

        # Risk-adjusted metrics
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)

        # Trading metrics
        trading_metrics = (
            self._calculate_trading_metrics(closed_positions) if closed_positions else {}
        )

        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(returns)

        # Combine all metrics
        all_metrics = {
            **core_metrics,
            **risk_metrics,
            **risk_adjusted_metrics,
            **trading_metrics,
            **advanced_metrics,
        }

        return all_metrics

    def _calculate_core_metrics(self, values: List[float], returns: pd.Series) -> Dict[str, float]:
        """Calculate core return metrics"""
        if not values or len(values) < 2:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "daily_return_mean": 0.0,
                "daily_return_std": 0.0,
            }

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Annualized return
        days = len(returns)
        if days > 0:
            annual_return = (1 + total_return) ** (252 / days) - 1
        else:
            annual_return = 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return_mean": returns.mean(),
            "daily_return_std": returns.std(),
        }

    def _calculate_risk_metrics(self, values: List[float], returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        if len(returns) == 0:
            return {
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "downside_deviation": 0.0,
            }

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Maximum drawdown
        max_drawdown = self.max_drawdown(values)

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)  # 5th percentile

        # Conditional Value at Risk (95%)
        cvar_95 = (
            returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        )

        # Downside deviation (volatility of negative returns)
        negative_returns = returns[returns < 0]
        downside_deviation = (
            negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        )

        return {
            "volatility": volatility,
            "max_drawdown": abs(max_drawdown),
            "var_95": abs(var_95),
            "cvar_95": abs(cvar_95),
            "downside_deviation": downside_deviation,
        }

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {"sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0}

        # Sharpe ratio
        sharpe_ratio = self.sharpe_ratio(returns, self.risk_free_rate)

        # Sortino ratio
        sortino_ratio = self.sortino_ratio(returns, self.risk_free_rate)

        # Calmar ratio
        max_dd = self.max_drawdown([0] + returns.cumsum().tolist())
        if abs(max_dd) > 0:
            annual_return = returns.mean() * 252
            calmar_ratio = annual_return / abs(max_dd)
        else:
            calmar_ratio = 0.0

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_trading_metrics(self, closed_positions: List) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        if not closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win_loss_ratio": 0.0,
                "avg_trade_return": 0.0,
                "avg_trade_duration": 0.0,
            }

        # Extract trade P&L
        trade_pnls = []
        trade_durations = []

        for position in closed_positions:
            if hasattr(position, "realized_pnl") and position.realized_pnl is not None:
                # Calculate return percentage
                cost_basis = position.shares * position.entry_price
                if cost_basis > 0:
                    trade_return = position.realized_pnl / cost_basis
                    trade_pnls.append(trade_return)

                # Calculate duration
                if hasattr(position, "days_held"):
                    trade_durations.append(position.days_held)

        if not trade_pnls:
            return {
                "total_trades": len(closed_positions),
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win_loss_ratio": 0.0,
                "avg_trade_return": 0.0,
                "avg_trade_duration": 0.0,
            }

        # Win rate
        winners = [pnl for pnl in trade_pnls if pnl > 0]
        losers = [pnl for pnl in trade_pnls if pnl < 0]
        win_rate = len(winners) / len(trade_pnls)

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average win/loss ratio
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = abs(np.mean(losers)) if losers else 1
        avg_win_loss_ratio = avg_winner / avg_loser if avg_loser > 0 else float("inf")

        return {
            "total_trades": len(trade_pnls),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "avg_trade_return": np.mean(trade_pnls),
            "avg_trade_duration": np.mean(trade_durations) if trade_durations else 0.0,
        }

    def _calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced metrics as specified in requirements.md"""
        if len(returns) == 0:
            return {
                "tail_ratio": 0.0,
                "omega_ratio": 0.0,
                "ulcer_index": 0.0,
                "recovery_time": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }

        # Tail ratio (95th percentile gain / 95th percentile loss)
        gain_95 = np.percentile(returns[returns > 0], 95) if len(returns[returns > 0]) > 0 else 0
        loss_95 = (
            abs(np.percentile(returns[returns < 0], 5)) if len(returns[returns < 0]) > 0 else 1
        )
        tail_ratio = gain_95 / loss_95 if loss_95 > 0 else 0

        # Omega ratio (probability weighted gain/loss ratio)
        omega_ratio = self.omega_ratio(returns)

        # Ulcer Index (root mean square of drawdowns)
        ulcer_index = self.ulcer_index(returns)

        # Recovery time (average time to recover from drawdown)
        recovery_time = self._calculate_avg_recovery_time(returns)

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        return {
            "tail_ratio": tail_ratio,
            "omega_ratio": omega_ratio,
            "ulcer_index": ulcer_index,
            "recovery_time": recovery_time,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio as specified in design.md"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sortino ratio (downside deviation version of Sharpe)"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            return float("inf")

        downside_std = negative_returns.std()
        if downside_std == 0:
            return 0.0

        return np.sqrt(252) * excess_returns.mean() / downside_std

    def max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown as specified in design.md"""
        if len(values) < 2:
            return 0.0

        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()  # Most negative value

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 1.0

        return gains / losses

    def ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index (root mean square of drawdowns)"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max

        # Root mean square of drawdowns
        return np.sqrt((drawdowns**2).mean())

    def _calculate_avg_recovery_time(self, returns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()

        recovery_times = []
        in_drawdown = False
        drawdown_start = 0

        for i, (value, peak) in enumerate(zip(cumulative, running_max)):
            if value < peak and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif value >= peak and in_drawdown:
                # Recovery complete
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False

        return np.mean(recovery_times) if recovery_times else 0.0

    def calculate_rolling_metrics(
        self, returns: pd.Series, window: int = 252
    ) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics"""
        if len(returns) < window:
            logger.warning(
                f"Insufficient data for rolling metrics (need {window}, have {len(returns)})"
            )
            return {}

        rolling_metrics = {}

        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: self.sharpe_ratio(pd.Series(x)), raw=False
        )
        rolling_metrics["rolling_sharpe"] = rolling_sharpe

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics["rolling_volatility"] = rolling_vol

        # Rolling maximum drawdown
        rolling_max_dd = returns.rolling(window).apply(
            lambda x: abs(self.max_drawdown((1 + pd.Series(x)).cumprod().tolist())), raw=False
        )
        rolling_metrics["rolling_max_drawdown"] = rolling_max_dd

        return rolling_metrics

    def compare_to_benchmark(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Compare portfolio performance to benchmark"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {}

        # Align series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return {}

        port_returns = returns.loc[common_index]
        bench_returns = benchmark_returns.loc[common_index]

        # Calculate metrics
        port_total_return = (1 + port_returns).prod() - 1
        bench_total_return = (1 + bench_returns).prod() - 1

        excess_returns = port_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)

        # Information ratio
        if tracking_error > 0:
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error
        else:
            information_ratio = 0.0

        # Beta calculation
        if bench_returns.std() > 0:
            beta = np.cov(port_returns, bench_returns)[0, 1] / bench_returns.var()
        else:
            beta = 0.0

        # Alpha calculation
        alpha = port_returns.mean() * 252 - (
            self.risk_free_rate + beta * (bench_returns.mean() * 252 - self.risk_free_rate)
        )

        return {
            "total_return": port_total_return,
            "benchmark_return": bench_total_return,
            "excess_return": port_total_return - bench_total_return,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "beta": beta,
            "alpha": alpha,
        }
