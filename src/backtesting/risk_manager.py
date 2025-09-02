"""
Risk Manager for portfolio risk management and position sizing
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management and position sizing following design.md specifications"""

    def __init__(self, risk_config: Dict[str, Any]):
        """
        Initialize risk manager with configuration

        Expected config structure:
        {
            "max_portfolio_risk": 0.2,      # 20% total portfolio risk
            "max_correlation": 0.7,         # Position correlation limit
            "max_sector_exposure": 0.3,     # 30% per sector limit
            "max_position_size": 0.1,       # 10% per position limit
            "var_limit": 0.05,              # 5% VaR limit
            "leverage_limit": 1.0,          # No leverage
            "concentration_limit": 0.25,    # 25% concentration limit
            "beta_limit": 1.5,              # Maximum portfolio beta
            "drawdown_limit": 0.15          # 15% max drawdown before stopping
        }
        """
        self.config = risk_config

        # Risk limits
        self.max_portfolio_risk = risk_config.get("max_portfolio_risk", 0.2)
        self.max_correlation = risk_config.get("max_correlation", 0.7)
        self.max_sector_exposure = risk_config.get("max_sector_exposure", 0.3)
        self.max_position_size = risk_config.get("max_position_size", 0.1)
        self.var_limit = risk_config.get("var_limit", 0.05)
        self.leverage_limit = risk_config.get("leverage_limit", 1.0)
        self.concentration_limit = risk_config.get("concentration_limit", 0.25)
        self.beta_limit = risk_config.get("beta_limit", 1.5)
        self.drawdown_limit = risk_config.get("drawdown_limit", 0.15)

        # Risk calculation parameters
        self.lookback_days = risk_config.get("lookback_days", 252)  # 1 year
        self.confidence_level = risk_config.get("confidence_level", 0.95)  # 95% VaR

        logger.info(
            f"RiskManager initialized with max portfolio risk: {self.max_portfolio_risk:.1%}"
        )

    def filter_signals(self, signals: List, portfolio, market_data: pd.Series) -> List:
        """
        Filter signals based on risk rules following exact design.md logic

        Args:
            signals: List of trading signals
            portfolio: Current portfolio state
            market_data: Current market data

        Returns:
            List of approved signals after risk filtering
        """
        approved_signals = []

        for signal in signals:
            try:
                # Size position based on risk rules
                sized_signal = self._size_position(signal, portfolio, market_data)

                if sized_signal and sized_signal.shares > 0:
                    # Check all risk limits
                    if self._check_all_risk_limits(sized_signal, portfolio, market_data):
                        approved_signals.append(sized_signal)
                    else:
                        logger.debug(f"Signal for {signal.ticker} rejected by risk limits")

            except Exception as e:
                logger.error(f"Error processing signal for {signal.ticker}: {e}")
                continue

        if len(approved_signals) < len(signals):
            logger.info(f"Risk manager approved {len(approved_signals)}/{len(signals)} signals")

        return approved_signals

    def _size_position(self, signal, portfolio, market_data: pd.Series):
        """Calculate optimal position size using Kelly criterion and risk limits"""
        try:
            # Get position sizing method from signal or use Kelly
            position_value = self._calculate_kelly_position_size(signal, portfolio, market_data)

            # Apply maximum position size limit
            max_position_value = portfolio.total_value * self.max_position_size
            position_value = min(position_value, max_position_value)

            # Convert to shares
            ticker_data = self._get_ticker_data(signal.ticker, market_data)
            share_price = self._get_price(ticker_data, "close")

            if share_price <= 0:
                logger.warning(f"Invalid share price for {signal.ticker}: {share_price}")
                return None

            shares = int(position_value / share_price)

            # Update signal with calculated position size
            signal.shares = shares

            return signal

        except Exception as e:
            logger.error(f"Error sizing position for {signal.ticker}: {e}")
            return None

    def _calculate_kelly_position_size(self, signal, portfolio, market_data: pd.Series) -> float:
        """
        Calculate optimal position size using Kelly criterion as specified in design.md

        Kelly fraction: f* = (bp - q) / b
        where b = win/loss ratio, p = win probability, q = loss probability
        """
        win_prob = signal.confidence
        expected_return = signal.expected_return

        # Use stop loss as potential loss
        potential_loss = self.config.get("stop_loss", 0.02)  # 2% default

        if potential_loss <= 0:
            potential_loss = 0.02  # Default 2%

        # Kelly fraction calculation
        win_loss_ratio = abs(expected_return) / potential_loss if potential_loss > 0 else 1.0
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Apply safety factor and constraints
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Cap at 10% as specified

        # Calculate position value
        position_value = portfolio.total_value * kelly_fraction

        return position_value

    def _check_all_risk_limits(self, signal, portfolio, market_data: pd.Series) -> bool:
        """Check if signal passes all risk limits"""

        # 1. Portfolio risk limit
        if not self._check_portfolio_risk_limit(signal, portfolio, market_data):
            return False

        # 2. Correlation limit
        if not self._check_correlation_limit(signal, portfolio, market_data):
            return False

        # 3. Sector exposure limit
        if not self._check_sector_exposure_limit(signal, portfolio, market_data):
            return False

        # 4. Position size limit (already applied in sizing)
        if not self._check_position_size_limit(signal, portfolio):
            return False

        # 5. VaR limit
        if not self._check_var_limit(signal, portfolio, market_data):
            return False

        # 6. Concentration limit
        if not self._check_concentration_limit(signal, portfolio):
            return False

        return True

    def _check_portfolio_risk_limit(self, signal, portfolio, market_data: pd.Series) -> bool:
        """Check portfolio risk limit"""
        try:
            current_risk = self.calculate_portfolio_risk(portfolio, market_data)

            # Estimate additional risk from new position
            ticker_data = self._get_ticker_data(signal.ticker, market_data)
            position_volatility = ticker_data.get("volatility", 0.02)  # Default 2%

            ticker_data = self._get_ticker_data(signal.ticker, market_data)
            share_price = self._get_price(ticker_data, "close")
            position_value = signal.shares * share_price
            position_weight = position_value / portfolio.total_value

            additional_risk = position_weight * position_volatility
            projected_risk = current_risk + additional_risk

            return projected_risk <= self.max_portfolio_risk

        except Exception as e:
            logger.error(f"Error checking portfolio risk limit: {e}")
            return False

    def _check_correlation_limit(self, signal, portfolio, market_data: pd.Series) -> bool:
        """Check position correlation limit"""
        if not portfolio.positions:
            return True  # No existing positions to correlate with

        # Simplified correlation check - in practice would use historical correlation matrix
        # For now, assume low correlation if different sectors or market caps

        return True  # Placeholder - implement with actual correlation data

    def _check_sector_exposure_limit(self, signal, portfolio, market_data: pd.Series) -> bool:
        """Check sector exposure limit"""
        # Get sector information (would need sector mapping data)
        ticker_data = self._get_ticker_data(signal.ticker, market_data)
        sector = ticker_data.get("sector", "Unknown")

        if sector == "Unknown":
            return True  # Can't check without sector data

        # Calculate current sector exposure
        sector_exposure = self._calculate_sector_exposure(portfolio, sector, {})

        # Calculate additional exposure from new position
        share_price = self._get_price(ticker_data, "close")
        position_value = signal.shares * share_price
        additional_exposure = position_value / portfolio.total_value

        total_sector_exposure = sector_exposure + additional_exposure

        return total_sector_exposure <= self.max_sector_exposure

    def _check_position_size_limit(self, signal, portfolio) -> bool:
        """Check individual position size limit"""
        ticker_data = self._get_ticker_data(signal.ticker, pd.Series())  # Simplified
        share_price = 100.0  # Placeholder - would get from market data
        position_value = signal.shares * share_price
        position_weight = position_value / portfolio.total_value

        return position_weight <= self.max_position_size

    def _check_var_limit(self, signal, portfolio, market_data: pd.Series) -> bool:
        """Check Value at Risk limit"""
        # Simplified VaR check - would implement full VaR calculation with correlation matrix
        return True  # Placeholder

    def _check_concentration_limit(self, signal, portfolio) -> bool:
        """Check concentration limit (largest position)"""
        ticker_data = self._get_ticker_data(signal.ticker, pd.Series())  # Simplified
        share_price = 100.0  # Placeholder
        position_value = signal.shares * share_price

        # Check if this would become the largest position
        if portfolio.positions:
            max_existing_value = max(pos.current_value for pos in portfolio.positions.values())
            max_total_value = max(position_value, max_existing_value)
        else:
            max_total_value = position_value

        concentration = max_total_value / portfolio.total_value

        return concentration <= self.concentration_limit

    def calculate_portfolio_risk(self, portfolio, market_data: pd.Series) -> float:
        """
        Calculate portfolio risk (volatility) as specified in design.md

        Returns:
            Portfolio volatility (annualized)
        """
        if not portfolio.positions:
            return 0.0

        try:
            # Get position weights
            total_value = portfolio.total_value
            weights = {}

            for ticker, position in portfolio.positions.items():
                if total_value > 0:
                    weights[ticker] = position.current_value / total_value
                else:
                    weights[ticker] = 0.0

            # Calculate weighted volatility (simplified - doesn't include correlations)
            portfolio_volatility = 0.0

            for ticker, weight in weights.items():
                ticker_data = self._get_ticker_data(ticker, market_data)
                ticker_volatility = ticker_data.get("volatility", 0.02)  # Default 2%
                portfolio_volatility += (weight**2) * (ticker_volatility**2)

            return np.sqrt(portfolio_volatility)  # Already annualized

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0

    def calculate_correlations(
        self, ticker: str, existing_tickers: List[str], market_data: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate correlations between ticker and existing positions

        Returns:
            Dictionary of correlations
        """
        correlations = {}

        # Placeholder - would implement with actual correlation calculation
        for existing_ticker in existing_tickers:
            if existing_ticker != ticker:
                # Simplified correlation estimate based on sector/market cap
                correlations[existing_ticker] = 0.3  # Default moderate correlation

        return correlations

    def _calculate_sector_exposure(
        self, portfolio, sector: str, sector_map: Dict[str, str]
    ) -> float:
        """Calculate current sector exposure"""
        sector_value = 0.0

        for ticker, position in portfolio.positions.items():
            position_sector = sector_map.get(ticker, "Unknown")
            if position_sector == sector:
                sector_value += position.current_value

        if portfolio.total_value > 0:
            return sector_value / portfolio.total_value
        else:
            return 0.0

    def _get_ticker_data(self, ticker: str, market_data: pd.Series) -> pd.Series:
        """Extract ticker-specific data from market data"""
        try:
            if isinstance(market_data.index, pd.MultiIndex):
                return market_data.loc[ticker]
            else:
                return market_data
        except (KeyError, IndexError):
            # Return empty series with default values
            return pd.Series(
                {"close": 100.0, "volatility": 0.02, "volume": 1000000, "sector": "Unknown"}
            )

    def _get_price(self, ticker_data: pd.Series, price_type: str = "close") -> float:
        """Get price from ticker data"""
        try:
            if price_type in ticker_data.index:
                return float(ticker_data[price_type])
            elif "close" in ticker_data.index:
                return float(ticker_data["close"])
            else:
                return 100.0  # Default price
        except (ValueError, TypeError):
            return 100.0  # Default price

    def calculate_position_beta(
        self, ticker: str, market_data: pd.Series, benchmark_data: pd.Series = None
    ) -> float:
        """Calculate position beta relative to benchmark"""
        # Placeholder - would implement actual beta calculation
        return 1.0

    def calculate_var(
        self, portfolio, confidence_level: float = 0.95, lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk for portfolio

        Returns:
            Dictionary with VaR metrics
        """
        if not portfolio.daily_returns or len(portfolio.daily_returns) < 30:
            return {"var_95": 0.0, "var_99": 0.0, "cvar_95": 0.0, "cvar_99": 0.0}

        returns = np.array(portfolio.daily_returns[-lookback_days:])

        # Calculate VaR at different confidence levels
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR

        # Calculate Conditional VaR (Expected Shortfall)
        cvar_95 = (
            returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        )
        cvar_99 = (
            returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        )

        return {
            "var_95": abs(var_95),  # Make positive for reporting
            "var_99": abs(var_99),
            "cvar_95": abs(cvar_95),
            "cvar_99": abs(cvar_99),
        }

    def check_drawdown_limit(self, portfolio) -> Dict[str, Any]:
        """
        Check if portfolio has breached drawdown limits

        Returns:
            Dictionary with drawdown status and metrics
        """
        if not portfolio.daily_values or len(portfolio.daily_values) < 2:
            return {
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
                "breach": False,
                "days_in_drawdown": 0,
            }

        values = np.array(portfolio.daily_values)

        # Calculate running maximum
        running_max = np.maximum.accumulate(values)

        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max

        current_drawdown = abs(drawdowns[-1])
        max_drawdown = abs(drawdowns.min())

        # Check for breach
        breach = current_drawdown > self.drawdown_limit

        # Calculate days in current drawdown
        days_in_drawdown = 0
        for i in range(len(drawdowns) - 1, -1, -1):
            if drawdowns[i] < -0.001:  # In drawdown (allowing for small rounding errors)
                days_in_drawdown += 1
            else:
                break

        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "breach": breach,
            "days_in_drawdown": days_in_drawdown,
            "limit": self.drawdown_limit,
        }

    def get_risk_report(self, portfolio, market_data: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        risk_metrics = {
            "portfolio_risk": self.calculate_portfolio_risk(portfolio, market_data),
            "var_metrics": self.calculate_var(portfolio),
            "drawdown_metrics": self.check_drawdown_limit(portfolio),
            "concentration": self._calculate_concentration(portfolio),
            "leverage": self._calculate_leverage(portfolio),
            "position_count": len(portfolio.positions),
            "cash_ratio": (
                portfolio.cash / portfolio.total_value if portfolio.total_value > 0 else 1.0
            ),
        }

        # Risk limit compliance
        compliance = {
            "portfolio_risk_ok": risk_metrics["portfolio_risk"] <= self.max_portfolio_risk,
            "var_ok": risk_metrics["var_metrics"]["var_95"] <= self.var_limit,
            "drawdown_ok": not risk_metrics["drawdown_metrics"]["breach"],
            "concentration_ok": risk_metrics["concentration"] <= self.concentration_limit,
            "leverage_ok": risk_metrics["leverage"] <= self.leverage_limit,
        }

        return {
            "metrics": risk_metrics,
            "compliance": compliance,
            "limits": {
                "max_portfolio_risk": self.max_portfolio_risk,
                "var_limit": self.var_limit,
                "drawdown_limit": self.drawdown_limit,
                "concentration_limit": self.concentration_limit,
                "leverage_limit": self.leverage_limit,
            },
        }

    def _calculate_concentration(self, portfolio) -> float:
        """Calculate portfolio concentration (largest position weight)"""
        if not portfolio.positions:
            return 0.0

        position_values = [pos.current_value for pos in portfolio.positions.values()]
        max_position = max(position_values) if position_values else 0.0

        if portfolio.total_value > 0:
            return max_position / portfolio.total_value
        else:
            return 0.0

    def _calculate_leverage(self, portfolio) -> float:
        """Calculate portfolio leverage ratio"""
        if portfolio.total_value <= 0:
            return 0.0

        gross_exposure = sum(pos.current_value for pos in portfolio.positions.values())
        return gross_exposure / portfolio.total_value
