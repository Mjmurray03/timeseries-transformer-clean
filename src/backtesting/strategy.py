"""
Trading Strategy implementation with signal generation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal"""

    ticker: str
    type: str  # 'BUY', 'SELL'
    shares: int
    confidence: float
    expected_return: float
    timestamp: Optional[datetime] = None
    reason: str = ""
    limit_price: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Strategy:
    """Trading strategy implementation following design.md specifications"""

    def __init__(self, strategy_params: Dict[str, Any]):
        """
        Initialize strategy with parameters

        Expected params structure:
        {
            "min_expected_return": 0.02,      # 2% minimum expected return
            "min_confidence": 0.7,            # 70% minimum confidence
            "max_positions": 10,              # Maximum concurrent positions
            "stop_loss": 0.02,                # 2% stop loss
            "profit_target": 0.05,            # 5% profit target
            "time_stop": 5,                   # 5 day time stop
            "exit_threshold": -0.01,          # Exit when prediction turns negative
            "position_sizing": "kelly",       # Position sizing method
            "rebalance_frequency": "daily"    # Rebalancing frequency
        }
        """
        self.config = strategy_params

        # Entry rules
        self.min_expected_return = strategy_params.get("min_expected_return", 0.02)
        self.min_confidence = strategy_params.get("min_confidence", 0.7)
        self.max_positions = strategy_params.get("max_positions", 10)

        # Exit rules
        self.stop_loss = strategy_params.get("stop_loss", 0.02)
        self.profit_target = strategy_params.get("profit_target", 0.05)
        self.time_stop = strategy_params.get("time_stop", 5)
        self.exit_threshold = strategy_params.get("exit_threshold", -0.01)

        # Strategy settings
        self.position_sizing = strategy_params.get("position_sizing", "fixed")
        self.base_position_size = strategy_params.get("position_size", 0.1)  # 10% default

        logger.info(
            f"Strategy initialized: min_return={self.min_expected_return:.2%}, "
            f"min_confidence={self.min_confidence:.2%}, max_positions={self.max_positions}"
        )

    def generate_signals(
        self, predictions: pd.Series, positions: Dict, historical_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """
        Generate trading signals from predictions following exact design.md structure

        Args:
            predictions: Daily predictions for all tickers
            positions: Current portfolio positions
            historical_data: Historical market data up to current date

        Returns:
            List of TradingSignal objects
        """
        signals = []

        for ticker in predictions.index:
            try:
                prediction = self._parse_prediction(predictions.loc[ticker])

                # Check entry conditions
                if self.should_enter(ticker, prediction, positions):
                    signal = self.create_entry_signal(ticker, prediction)
                    signals.append(signal)
                    logger.debug(
                        f"Entry signal: {signal.type} {signal.shares} shares of {ticker} "
                        f"(expected return: {prediction['return_5d']:.2%})"
                    )

                # Check exit conditions for existing positions
                elif ticker in positions:
                    if self.should_exit(ticker, positions[ticker], prediction):
                        signal = self.create_exit_signal(ticker, positions[ticker])
                        signals.append(signal)
                        logger.debug(
                            f"Exit signal: {signal.type} {signal.shares} shares of {ticker}"
                        )

            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {e}")
                continue

        return signals

    def _parse_prediction(self, prediction_data) -> Dict[str, float]:
        """Parse prediction data into standardized format"""
        if isinstance(prediction_data, dict):
            return prediction_data
        elif isinstance(prediction_data, pd.Series):
            return {
                "return_5d": prediction_data.get(
                    "return_5d", prediction_data.iloc[0] if len(prediction_data) > 0 else 0.0
                ),
                "confidence": prediction_data.get("confidence", 0.5),
                "volatility": prediction_data.get("volatility", 0.02),
            }
        else:
            # Single value - assume it's the expected return
            return {
                "return_5d": float(prediction_data),
                "confidence": 0.5,  # Default confidence
                "volatility": 0.02,  # Default volatility
            }

    def should_enter(self, ticker: str, prediction: Dict[str, float], positions: Dict) -> bool:
        """
        Determine if should enter position following exact design.md logic

        Args:
            ticker: Stock ticker
            prediction: Prediction dictionary with return_5d, confidence, etc.
            positions: Current positions dictionary

        Returns:
            Boolean indicating whether to enter position
        """
        # Check if already in position
        if ticker in positions:
            return False

        # Check prediction threshold
        expected_return = prediction["return_5d"]
        if expected_return < self.min_expected_return:
            return False

        # Check confidence threshold
        confidence = prediction["confidence"]
        if confidence < self.min_confidence:
            return False

        # Check portfolio limits
        if len(positions) >= self.max_positions:
            return False

        return True

    def should_exit(self, ticker: str, position, prediction: Dict[str, float]) -> bool:
        """
        Determine if should exit position following exact design.md logic

        Args:
            ticker: Stock ticker
            position: Position object
            prediction: Current prediction

        Returns:
            Boolean indicating whether to exit position
        """
        # Check stop loss
        if hasattr(position, "unrealized_pnl_pct"):
            if position.unrealized_pnl_pct <= -self.stop_loss:
                return True

        # Check profit target
        if hasattr(position, "unrealized_pnl_pct"):
            if position.unrealized_pnl_pct >= self.profit_target:
                return True

        # Check time stop
        if hasattr(position, "days_held"):
            if position.days_held >= self.time_stop:
                return True

        # Check prediction reversal
        if prediction["return_5d"] < self.exit_threshold:
            return True

        return False

    def create_entry_signal(self, ticker: str, prediction: Dict[str, float]) -> TradingSignal:
        """Create entry signal with proper position sizing"""
        expected_return = prediction["return_5d"]
        confidence = prediction["confidence"]

        # Calculate position size based on strategy
        position_size_pct = self._calculate_position_size(prediction)

        # For now, use a default number of shares (will be adjusted by risk manager)
        shares = 100  # This will be properly sized by the risk manager

        signal = TradingSignal(
            ticker=ticker,
            type="BUY",
            shares=shares,
            confidence=confidence,
            expected_return=expected_return,
            reason=f"Expected return: {expected_return:.2%}, Confidence: {confidence:.2%}",
        )

        return signal

    def create_exit_signal(self, ticker: str, position) -> TradingSignal:
        """Create exit signal for existing position"""
        shares = position.shares

        signal = TradingSignal(
            ticker=ticker,
            type="SELL",
            shares=shares,
            confidence=1.0,  # Exit signals have high confidence
            expected_return=0.0,  # Not applicable for exits
            reason=f"Exit condition met for {ticker}",
        )

        return signal

    def _calculate_position_size(self, prediction: Dict[str, float]) -> float:
        """
        Calculate position size as percentage of portfolio

        Args:
            prediction: Prediction with expected return, confidence, volatility

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        if self.position_sizing == "fixed":
            return self.base_position_size

        elif self.position_sizing == "kelly":
            # Kelly criterion: f* = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            win_prob = prediction["confidence"]
            expected_return = prediction["return_5d"]

            # Assume symmetric loss of stop_loss amount
            avg_win = expected_return
            avg_loss = self.stop_loss

            if avg_loss > 0:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win

                # Apply safety factor and cap
                kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.2))  # Cap at 20%
                return kelly_fraction
            else:
                return self.base_position_size

        elif self.position_sizing == "volatility":
            # Inverse volatility sizing
            volatility = prediction.get("volatility", 0.02)
            target_volatility = 0.02  # 2% target vol

            if volatility > 0:
                vol_adjusted_size = self.base_position_size * (target_volatility / volatility)
                return max(0.01, min(vol_adjusted_size, 0.2))  # 1% to 20% range
            else:
                return self.base_position_size

        else:
            return self.base_position_size

    def update_strategy_params(self, new_params: Dict[str, Any]):
        """Update strategy parameters dynamically"""
        for key, value in new_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated strategy parameter {key} to {value}")
            else:
                logger.warning(f"Unknown strategy parameter: {key}")

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy configuration"""
        return {
            "min_expected_return": self.min_expected_return,
            "min_confidence": self.min_confidence,
            "max_positions": self.max_positions,
            "stop_loss": self.stop_loss,
            "profit_target": self.profit_target,
            "time_stop": self.time_stop,
            "exit_threshold": self.exit_threshold,
            "position_sizing": self.position_sizing,
            "base_position_size": self.base_position_size,
        }

    def backtest_signal_performance(
        self, signals: List[TradingSignal], actual_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze historical signal performance

        Args:
            signals: Historical signals generated
            actual_returns: Actual market returns data

        Returns:
            Signal performance metrics
        """
        if not signals:
            return {}

        correct_signals = 0
        total_signals = len(signals)

        for signal in signals:
            if signal.ticker in actual_returns.columns:
                # Get actual return following signal
                signal_date = signal.timestamp.date() if signal.timestamp else None
                if signal_date and signal_date in actual_returns.index:
                    actual_return = actual_returns.loc[signal_date, signal.ticker]

                    # Check if signal direction was correct
                    if signal.type == "BUY" and actual_return > 0:
                        correct_signals += 1
                    elif signal.type == "SELL" and actual_return < 0:
                        correct_signals += 1

        accuracy = correct_signals / total_signals if total_signals > 0 else 0.0

        return {
            "total_signals": total_signals,
            "correct_signals": correct_signals,
            "signal_accuracy": accuracy,
            "buy_signals": len([s for s in signals if s.type == "BUY"]),
            "sell_signals": len([s for s in signals if s.type == "SELL"]),
            "avg_confidence": np.mean([s.confidence for s in signals]),
            "avg_expected_return": np.mean([s.expected_return for s in signals]),
        }
