"""
Market Simulator with realistic order execution, transaction costs, and slippage
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutedTrade:
    """Represents an executed trade with all costs"""

    ticker: str
    type: str  # 'BUY' or 'SELL'
    shares: int
    execution_price: float
    commission: float
    slippage: float
    spread_cost: float
    timestamp: datetime

    @property
    def total_cost(self) -> float:
        """Total cost including all fees (for BUY orders)"""
        if self.type == "BUY":
            return self.shares * self.execution_price + self.commission
        else:
            return self.shares * self.execution_price - self.commission


@dataclass
class OrderExecution:
    """Represents an order for execution"""

    ticker: str
    type: str  # 'BUY', 'SELL', 'LIMIT_BUY', 'LIMIT_SELL'
    shares: int
    limit_price: Optional[float] = None  # For limit orders
    timestamp: Optional[datetime] = None


class CostModel:
    """Transaction cost model implementation following exact specifications"""

    def __init__(self, cost_config: Dict[str, Any]):
        """
        Initialize cost model with configuration

        Expected config structure from requirements.md:
        {
            "commission": {
                "fixed": 1.0,           # $1 per trade
                "percentage": 0.001     # 0.1% of trade value
            },
            "spread": {
                "base": 0.0001,         # 1 basis point
                "size_factor": 0.00001  # Increases with size
            },
            "slippage": {
                "base": 0.0005,         # 5 basis points
                "volatility_factor": 0.1, # Scales with volatility
                "size_impact": 0.0001   # Linear impact
            },
            "market_impact": {
                "temporary": 0.0002,    # Temporary impact
                "permanent": 0.0001     # Permanent impact
            }
        }
        """
        self.commission = cost_config.get("commission", {"fixed": 1.0, "percentage": 0.001})
        self.spread = cost_config.get("spread", {"base": 0.0001, "size_factor": 0.00001})
        self.slippage = cost_config.get(
            "slippage", {"base": 0.0005, "volatility_factor": 0.1, "size_impact": 0.0001}
        )
        self.market_impact = cost_config.get(
            "market_impact", {"temporary": 0.0002, "permanent": 0.0001}
        )

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission based on fixed + percentage model"""
        fixed_cost = self.commission["fixed"]
        percentage_cost = trade_value * self.commission["percentage"]
        return fixed_cost + percentage_cost

    def calculate_spread_cost(self, ticker: str, shares: int, market_data: pd.Series) -> float:
        """Calculate bid-ask spread cost"""
        # If bid/ask data is available, use it; otherwise estimate
        if "bid" in market_data.index and "ask" in market_data.index:
            bid = market_data["bid"]
            ask = market_data["ask"]
            spread = ask - bid
        else:
            # Estimate spread based on price and volatility
            price = self._get_price(market_data, "close")
            volatility = market_data.get("volatility", 0.02)  # Default 2% volatility
            spread = price * (self.spread["base"] + volatility * 0.1)

        # Size impact on spread
        size_factor = abs(shares) * self.spread["size_factor"]
        adjusted_spread = spread * (1 + size_factor)

        # Cost is half the spread (assuming we cross the spread)
        return adjusted_spread * abs(shares) * 0.5

    def calculate_slippage(
        self, ticker: str, shares: int, order_type: str, market_data: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate slippage with exact formulas from Market Simulator section
        Implements fixed, linear, and square-root slippage models as specified
        """
        price = self._get_price(market_data, "close")
        volatility = market_data.get("volatility", 0.02)
        volume = market_data.get("volume", 1000000)  # Default volume

        # Base slippage (fixed component)
        base_slippage_pct = self.slippage["base"]

        # Volatility adjustment (linear component)
        volatility_adjustment = volatility * self.slippage["volatility_factor"]

        # Size impact (square-root model for market impact)
        trade_value = abs(shares) * price
        market_cap = market_data.get("market_cap", 1e9)  # Default $1B market cap

        # Square-root size impact as specified
        size_impact_pct = self.slippage["size_impact"] * np.sqrt(trade_value / market_cap)

        # Temporary vs permanent impact
        temporary_impact = self.market_impact["temporary"] * np.sqrt(abs(shares) / volume)
        permanent_impact = self.market_impact["permanent"] * (abs(shares) / volume)

        # Total slippage percentage
        total_slippage_pct = (
            base_slippage_pct
            + volatility_adjustment
            + size_impact_pct
            + temporary_impact
            + permanent_impact
        )

        # Direction matters: worse price for both buy and sell
        if order_type in ["BUY", "LIMIT_BUY"]:
            slippage_price = price * (1 + total_slippage_pct)
        else:  # SELL, LIMIT_SELL
            slippage_price = price * (1 - total_slippage_pct)

        return {
            "slippage_pct": total_slippage_pct,
            "slippage_price": slippage_price,
            "base_slippage": base_slippage_pct,
            "volatility_impact": volatility_adjustment,
            "size_impact": size_impact_pct,
            "temporary_impact": temporary_impact,
            "permanent_impact": permanent_impact,
        }

    def _get_price(self, market_data: pd.Series, price_type: str = "close") -> float:
        """Extract price from market data, handling different formats"""
        if price_type in market_data.index:
            return market_data[price_type]
        elif "close" in market_data.index:
            return market_data["close"]
        elif "Close" in market_data.index:
            return market_data["Close"]
        else:
            # If it's a single value, return it
            if isinstance(market_data, (int, float)):
                return float(market_data)
            # Otherwise, take the first numeric value
            return float(market_data.iloc[0])


class MarketSimulator:
    """Simulates market execution with realistic costs and slippage"""

    def __init__(self, market_config: Dict[str, Any]):
        """
        Initialize market simulator

        Expected config:
        {
            "cost_model": {...},  # CostModel configuration
            "liquidity_model": {...},  # Liquidity constraints
            "execution_delay": 0,  # Execution delay in minutes
            "market_hours": {"start": "09:30", "end": "16:00"}
        }
        """
        self.cost_model = CostModel(market_config.get("cost_model", {}))
        self.liquidity_model = market_config.get("liquidity_model", {})
        self.execution_delay = market_config.get("execution_delay", 0)
        self.market_hours = market_config.get("market_hours", {"start": "09:30", "end": "16:00"})

        logger.info("MarketSimulator initialized with realistic execution model")

    def execute_order_with_slippage(
        self, order: OrderExecution, market_data: pd.Series
    ) -> ExecutedTrade:
        """
        Execute single order with realistic slippage and transaction costs

        Args:
            order: Order to execute
            market_data: Market data for the ticker

        Returns:
            ExecutedTrade with all costs calculated
        """
        try:
            # Get base price
            base_price = self.cost_model._get_price(market_data, "close")

            # Calculate slippage
            slippage_data = self.cost_model.calculate_slippage(
                order.ticker, order.shares, order.type, market_data
            )
            execution_price = slippage_data["slippage_price"]
            slippage_cost = abs(execution_price - base_price) * abs(order.shares)

            # Calculate spread cost
            spread_cost = self.cost_model.calculate_spread_cost(
                order.ticker, order.shares, market_data
            )

            # Calculate commission
            trade_value = abs(order.shares) * execution_price
            commission = self.cost_model.calculate_commission(trade_value)

            # Handle limit orders
            if order.type in ["LIMIT_BUY", "LIMIT_SELL"] and order.limit_price:
                if order.type == "LIMIT_BUY" and execution_price > order.limit_price:
                    # Limit buy price too low - partial fill or no fill
                    execution_price = order.limit_price
                elif order.type == "LIMIT_SELL" and execution_price < order.limit_price:
                    # Limit sell price too high - partial fill or no fill
                    execution_price = order.limit_price

            # Create executed trade
            executed_trade = ExecutedTrade(
                ticker=order.ticker,
                type=order.type,
                shares=order.shares,
                execution_price=execution_price,
                commission=commission,
                slippage=slippage_cost,
                spread_cost=spread_cost,
                timestamp=order.timestamp or datetime.now(),
            )

            logger.debug(
                f"Executed {order.type} {order.shares} {order.ticker} @ ${execution_price:.2f} "
                f"(commission: ${commission:.2f}, slippage: ${slippage_cost:.2f})"
            )

            return executed_trade

        except Exception as e:
            logger.error(f"Error executing order for {order.ticker}: {e}")
            raise

    def calculate_transaction_costs(
        self, order: OrderExecution, market_data: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate comprehensive transaction costs for an order

        Returns:
            Dictionary with breakdown of all costs
        """
        try:
            base_price = self.cost_model._get_price(market_data, "close")
            trade_value = abs(order.shares) * base_price

            # Commission
            commission = self.cost_model.calculate_commission(trade_value)

            # Slippage breakdown
            slippage_data = self.cost_model.calculate_slippage(
                order.ticker, order.shares, order.type, market_data
            )

            # Spread cost
            spread_cost = self.cost_model.calculate_spread_cost(
                order.ticker, order.shares, market_data
            )

            # Market impact components
            total_slippage_cost = abs(slippage_data["slippage_price"] - base_price) * abs(
                order.shares
            )

            total_cost = commission + total_slippage_cost + spread_cost

            return {
                "commission": commission,
                "slippage": total_slippage_cost,
                "slippage_pct": slippage_data["slippage_pct"],
                "spread_cost": spread_cost,
                "market_impact": {
                    "temporary": slippage_data["temporary_impact"] * trade_value,
                    "permanent": slippage_data["permanent_impact"] * trade_value,
                    "size_impact": slippage_data["size_impact"] * trade_value,
                },
                "total_cost": total_cost,
                "cost_as_pct_of_trade": total_cost / trade_value if trade_value > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Error calculating transaction costs for {order.ticker}: {e}")
            return {"total_cost": 0, "error": str(e)}

    def execute_orders(
        self, signals: List, market_data: pd.Series, portfolio
    ) -> List[ExecutedTrade]:
        """
        Execute orders with realistic costs following exact OrderExecution class design

        Args:
            signals: List of trading signals (OrderExecution objects)
            market_data: Market data for current time
            portfolio: Current portfolio state

        Returns:
            List of ExecutedTrade objects
        """
        executed_trades = []

        for signal in signals:
            try:
                executed_trade = self._execute_single_order(signal, market_data, portfolio)
                if executed_trade:
                    executed_trades.append(executed_trade)
            except Exception as e:
                logger.error(f"Failed to execute order for {signal.ticker}: {e}")
                continue

        return executed_trades

    def _execute_single_order(
        self, signal, market_data: pd.Series, portfolio
    ) -> Optional[ExecutedTrade]:
        """Execute a single order with all cost calculations"""
        ticker = signal.ticker

        # Get ticker-specific market data
        if isinstance(market_data.index, pd.MultiIndex):
            # Handle MultiIndex case
            try:
                ticker_data = market_data.loc[ticker]
            except KeyError:
                logger.warning(f"No market data for {ticker}")
                return None
        else:
            # Assume single ticker data
            ticker_data = market_data

        # Get base execution price
        if hasattr(signal, "type"):
            order_type = signal.type
        else:
            order_type = "BUY" if signal.shares > 0 else "SELL"

        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(signal, ticker_data, order_type)

        # Calculate all transaction costs
        trade_value = abs(signal.shares) * execution_price

        # Commission
        commission = self.cost_model.calculate_commission(trade_value)

        # Spread cost
        spread_cost = self.cost_model.calculate_spread_cost(ticker, signal.shares, ticker_data)

        # Slippage (already incorporated in execution_price, but track separately)
        slippage_info = self.cost_model.calculate_slippage(
            ticker, signal.shares, order_type, ticker_data
        )
        base_price = self.cost_model._get_price(ticker_data, "close")
        slippage_cost = abs(execution_price - base_price) * abs(signal.shares)

        # Check liquidity constraints
        if not self._check_liquidity(signal, ticker_data):
            logger.warning(f"Insufficient liquidity for {ticker}, skipping order")
            return None

        # Check capital requirements for BUY orders
        if order_type in ["BUY", "LIMIT_BUY"]:
            required_capital = execution_price * signal.shares + commission
            if required_capital > portfolio.cash:
                logger.warning(
                    f"Insufficient cash for {ticker} purchase, required: ${required_capital:.2f}, available: ${portfolio.cash:.2f}"
                )
                return None

        # Create executed trade
        executed_trade = ExecutedTrade(
            ticker=ticker,
            type=order_type,
            shares=abs(signal.shares),  # Always positive
            execution_price=execution_price,
            commission=commission,
            slippage=slippage_cost,
            spread_cost=spread_cost,
            timestamp=getattr(signal, "timestamp", datetime.now()),
        )

        logger.debug(
            f"Executed {order_type} order: {executed_trade.shares} shares of {ticker} "
            f"at ${execution_price:.2f} (commission: ${commission:.2f}, slippage: ${slippage_cost:.2f})"
        )

        return executed_trade

    def _calculate_execution_price(self, signal, market_data: pd.Series, order_type: str) -> float:
        """Calculate execution price with slippage using exact formulas"""
        base_price = self.cost_model._get_price(market_data, "close")

        # For limit orders, check if limit price is favorable
        if hasattr(signal, "limit_price") and signal.limit_price is not None:
            if order_type == "LIMIT_BUY" and signal.limit_price < base_price:
                # Limit buy below market - use limit price
                base_price = signal.limit_price
            elif order_type == "LIMIT_SELL" and signal.limit_price > base_price:
                # Limit sell above market - use limit price
                base_price = signal.limit_price
            else:
                # Limit order not favorable, might not execute
                # For simulation, we'll execute at market with slippage
                pass

        # Apply slippage
        slippage_info = self.cost_model.calculate_slippage(
            signal.ticker, signal.shares, order_type, market_data
        )

        return slippage_info["slippage_price"]

    def _check_liquidity(self, signal, market_data: pd.Series) -> bool:
        """Check if order can be executed given liquidity constraints"""
        volume = market_data.get("volume", 1000000)  # Default volume

        # Simple liquidity check: don't allow orders larger than 10% of daily volume
        max_shares = volume * 0.1

        if abs(signal.shares) > max_shares:
            return False

        return True

    def calculate_market_impact(
        self, ticker: str, shares: int, market_data: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate detailed market impact breakdown

        Returns:
            Dictionary with impact components
        """
        price = self.cost_model._get_price(market_data, "close")
        volume = market_data.get("volume", 1000000)

        trade_value = abs(shares) * price
        volume_ratio = abs(shares) / volume

        # Temporary impact (mean-reverting)
        temporary_impact = self.cost_model.market_impact["temporary"] * np.sqrt(volume_ratio)

        # Permanent impact (informational)
        permanent_impact = self.cost_model.market_impact["permanent"] * volume_ratio

        # Total impact in basis points
        total_impact_bp = (temporary_impact + permanent_impact) * 10000

        return {
            "temporary_impact": temporary_impact,
            "permanent_impact": permanent_impact,
            "total_impact_bp": total_impact_bp,
            "volume_ratio": volume_ratio,
            "trade_value": trade_value,
        }

    def simulate_partial_fills(self, signal, market_data: pd.Series) -> List[ExecutedTrade]:
        """
        Simulate partial fills for large orders

        Returns:
            List of partial fills
        """
        volume = market_data.get("volume", 1000000)
        max_fill_ratio = 0.05  # Max 5% of volume per fill
        max_shares_per_fill = int(volume * max_fill_ratio)

        remaining_shares = abs(signal.shares)
        fills = []
        fill_number = 0

        while remaining_shares > 0:
            fill_shares = min(remaining_shares, max_shares_per_fill)

            # Create partial fill signal
            partial_signal = OrderExecution(
                ticker=signal.ticker,
                type=signal.type,
                shares=fill_shares,
                timestamp=signal.timestamp,
            )

            # Calculate execution with increasing slippage
            slippage_multiplier = 1 + (fill_number * 0.1)  # 10% more slippage per fill

            # This would need full execution logic - simplified here
            fills.append(partial_signal)

            remaining_shares -= fill_shares
            fill_number += 1

            # Safety break
            if fill_number > 10:
                break

        return fills

    def get_execution_summary(self, executed_trades: List[ExecutedTrade]) -> Dict[str, Any]:
        """Generate execution summary statistics"""
        if not executed_trades:
            return {}

        total_trades = len(executed_trades)
        total_volume = sum(trade.shares for trade in executed_trades)
        total_value = sum(trade.shares * trade.execution_price for trade in executed_trades)
        total_commission = sum(trade.commission for trade in executed_trades)
        total_slippage = sum(trade.slippage for trade in executed_trades)
        total_spread_cost = sum(trade.spread_cost for trade in executed_trades)

        return {
            "total_trades": total_trades,
            "total_volume": total_volume,
            "total_value": total_value,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_spread_cost": total_spread_cost,
            "total_costs": total_commission + total_slippage + total_spread_cost,
            "avg_trade_value": total_value / total_trades,
            "cost_as_pct_of_value": (total_commission + total_slippage + total_spread_cost)
            / total_value
            * 100,
            "buy_trades": len([t for t in executed_trades if t.type == "BUY"]),
            "sell_trades": len([t for t in executed_trades if t.type == "SELL"]),
        }
