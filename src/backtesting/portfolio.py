"""
Portfolio management and position tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single position in the portfolio"""

    ticker: str
    entry_price: float
    shares: int
    entry_date: datetime
    commission: float = 0.0
    current_price: float = 0.0

    # Optional fields for closed positions
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    realized_pnl: Optional[float] = None

    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price

    @property
    def current_value(self) -> float:
        """Current market value of position"""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis including commission"""
        return self.shares * self.entry_price + self.commission

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage"""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    @property
    def days_held(self) -> int:
        """Number of days position has been held"""
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        else:
            return (datetime.now() - self.entry_date).days

    def update_price(self, new_price: float):
        """Update current market price"""
        self.current_price = new_price

    def close(self, exit_price: float, exit_date: datetime, commission: float = 0.0) -> float:
        """
        Close position and calculate realized P&L

        Returns:
            Realized P&L including all commissions
        """
        self.exit_price = exit_price
        self.exit_date = exit_date

        # Calculate realized P&L (subtract both entry and exit commissions)
        self.realized_pnl = (
            (exit_price - self.entry_price) * self.shares - self.commission - commission
        )

        return self.realized_pnl


class Portfolio:
    """Portfolio state and management"""

    def __init__(self, initial_capital: float):
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict[str, Any]] = []
        self.initial_capital = initial_capital
        self.closed_positions: List[Position] = []

        # Performance tracking
        self.daily_values: List[float] = []
        self.daily_returns: List[float] = []

        logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")

    def update(self, trades: List, market_prices: pd.Series):
        """Update portfolio with executed trades and current market prices"""
        # Process trades
        for trade in trades:
            if trade.type == "BUY":
                self._open_position(trade, market_prices)
            elif trade.type == "SELL":
                self._close_position(trade, market_prices)

        # Update all position prices
        self._mark_to_market(market_prices)

        # Record current state
        self._record_snapshot()

    def _open_position(self, trade, market_prices: pd.Series):
        """Open new position"""
        if trade.ticker in self.positions:
            logger.warning(f"Already have position in {trade.ticker}, combining positions")
            # For simplicity, we'll replace the existing position
            # In a real system, you'd want to handle position combining properly

        position = Position(
            ticker=trade.ticker,
            entry_price=trade.execution_price,
            shares=trade.shares,
            entry_date=trade.timestamp,
            commission=trade.commission,
            current_price=trade.execution_price,
        )

        self.positions[trade.ticker] = position
        self.cash -= trade.total_cost

        logger.debug(
            f"Opened position: {trade.shares} shares of {trade.ticker} at ${trade.execution_price:.2f}"
        )

    def _close_position(self, trade, market_prices: pd.Series):
        """Close existing position"""
        if trade.ticker not in self.positions:
            logger.warning(f"Attempting to close non-existent position in {trade.ticker}")
            return

        position = self.positions[trade.ticker]

        # Calculate realized P&L
        realized_pnl = position.close(trade.execution_price, trade.timestamp, trade.commission)

        # Add cash from sale (minus commission)
        self.cash += trade.execution_price * position.shares - trade.commission

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[trade.ticker]

        logger.debug(
            f"Closed position: {position.shares} shares of {trade.ticker} "
            f"at ${trade.execution_price:.2f}, P&L: ${realized_pnl:.2f}"
        )

    def _mark_to_market(self, market_prices: pd.Series):
        """Update position values with current market prices"""
        for ticker, position in self.positions.items():
            if ticker in market_prices.index:
                # Handle MultiIndex case
                if isinstance(market_prices.index, pd.MultiIndex):
                    # Assume the price is in a 'close' column or similar
                    if hasattr(market_prices.loc[ticker], "close"):
                        position.update_price(market_prices.loc[ticker]["close"])
                    else:
                        # Use the first numeric column
                        price_data = market_prices.loc[ticker]
                        if isinstance(price_data, pd.Series):
                            position.update_price(
                                price_data.iloc[0]
                                if len(price_data) > 0
                                else position.current_price
                            )
                        else:
                            position.update_price(price_data)
                else:
                    position.update_price(market_prices.loc[ticker])
            else:
                logger.warning(f"No market price available for {ticker}")

    def _record_snapshot(self):
        """Record current portfolio state"""
        snapshot = self.get_snapshot()
        self.history.append(snapshot)

        # Track daily values and returns
        current_value = snapshot["total_value"]
        self.daily_values.append(current_value)

        if len(self.daily_values) > 1:
            daily_return = (current_value - self.daily_values[-2]) / self.daily_values[-2]
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get current portfolio snapshot"""
        return {
            "timestamp": datetime.now(),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "num_positions": len(self.positions),
            "unrealized_pnl": self.unrealized_pnl,
            "total_return": self.total_return,
            "positions": {
                ticker: {
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "current_value": pos.current_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "days_held": pos.days_held,
                }
                for ticker, pos in self.positions.items()
            },
        }

    @property
    def positions_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.current_value for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed positions"""
        return sum(
            pos.realized_pnl for pos in self.closed_positions if pos.realized_pnl is not None
        )

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return(self) -> float:
        """Total portfolio return since inception"""
        return (self.total_value - self.initial_capital) / self.initial_capital

    @property
    def daily_return(self) -> float:
        """Most recent daily return"""
        return self.daily_returns[-1] if self.daily_returns else 0.0

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker"""
        return self.positions.get(ticker)

    def has_position(self, ticker: str) -> bool:
        """Check if portfolio has position in ticker"""
        return ticker in self.positions

    def get_exposure(self, ticker: str) -> float:
        """Get exposure to a specific ticker as fraction of total value"""
        if ticker not in self.positions:
            return 0.0
        return self.positions[ticker].current_value / self.total_value

    def get_sector_exposure(self, sector_map: Dict[str, str]) -> Dict[str, float]:
        """
        Get sector exposure breakdown

        Args:
            sector_map: Dictionary mapping tickers to sectors

        Returns:
            Dictionary of sector exposures as fractions
        """
        sector_values = {}

        for ticker, position in self.positions.items():
            sector = sector_map.get(ticker, "Unknown")
            if sector not in sector_values:
                sector_values[sector] = 0.0
            sector_values[sector] += position.current_value

        # Convert to fractions
        if self.total_value > 0:
            return {sector: value / self.total_value for sector, value in sector_values.items()}
        else:
            return sector_values

    def calculate_portfolio_risk(
        self, returns_data: pd.DataFrame, lookback_days: int = 30
    ) -> float:
        """
        Calculate portfolio risk (volatility)

        Args:
            returns_data: Historical returns data for all tickers
            lookback_days: Number of days to look back for volatility calculation

        Returns:
            Portfolio volatility (annualized)
        """
        if not self.positions:
            return 0.0

        # Get weights
        total_value = self.total_value
        weights = {}
        for ticker, position in self.positions.items():
            if total_value > 0:
                weights[ticker] = position.current_value / total_value
            else:
                weights[ticker] = 0.0

        # Calculate portfolio volatility
        portfolio_returns = []
        for ticker, weight in weights.items():
            if ticker in returns_data.columns:
                ticker_returns = returns_data[ticker].tail(lookback_days)
                portfolio_returns.append(ticker_returns * weight)

        if portfolio_returns:
            portfolio_return_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            return portfolio_return_series.std() * np.sqrt(252)  # Annualized
        else:
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "initial_capital": self.initial_capital,
            "current_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_return": self.total_return,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "num_positions": len(self.positions),
            "num_closed_positions": len(self.closed_positions),
            "avg_daily_return": np.mean(self.daily_returns) if self.daily_returns else 0.0,
            "volatility": np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0.0,
            "sharpe_ratio": (
                (np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252))
                if self.daily_returns and np.std(self.daily_returns) > 0
                else 0.0
            ),
        }
