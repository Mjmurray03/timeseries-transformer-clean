#!/usr/bin/env python3
"""
# COMPONENT: ML Threshold Strategy
# PURPOSE: Convert ML model predictions to trading signals with configurable thresholds
# INPUTS: Model predictions with return forecasts and confidence scores
# OUTPUTS: Trading signals with position sizing and risk parameters
# VERIFICATION: Signal validation, position limits, risk constraints
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..strategy import Strategy, TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Structured ML prediction data"""

    ticker: str
    predicted_return_1d: float
    predicted_return_3d: float
    predicted_return_5d: float
    confidence: float
    volatility: float
    timestamp: datetime

    @property
    def primary_return(self) -> float:
        """Primary return forecast (3-day by default)"""
        return self.predicted_return_3d

    @property
    def risk_adjusted_return(self) -> float:
        """Return adjusted by confidence and volatility"""
        return self.primary_return * self.confidence / max(self.volatility, 0.01)


class MLThresholdStrategy(Strategy):
    """
    # COMPONENT: ML Threshold Trading Strategy
    # PURPOSE: Generate trading signals from ML predictions with multiple thresholds
    # INPUTS: ML predictions, market data, portfolio state
    # OUTPUTS: Filtered and sized trading signals
    # VERIFICATION: Threshold validation, position sizing, correlation limits
    """

    def __init__(
        self,
        return_threshold: float = 0.02,
        confidence_threshold: float = 0.7,
        max_positions: int = 5,
        position_sizing: str = "kelly",
        correlation_threshold: float = 0.7,
        volatility_target: float = 0.15,
        max_position_size: float = 0.2,
        min_holding_period: int = 1,
        max_holding_period: int = 10,
        rebalance_threshold: float = 0.05,
        **kwargs,
    ):
        """
        Initialize ML-based trading strategy

        Args:
            return_threshold: Minimum expected return to enter position (2%)
            confidence_threshold: Minimum confidence score to enter (0.7)
            max_positions: Maximum concurrent positions (5)
            position_sizing: Method for position sizing ('kelly', 'equal_weight', 'volatility_scaled')
            correlation_threshold: Maximum correlation between positions (0.7)
            volatility_target: Target portfolio volatility (15%)
            max_position_size: Maximum single position size (20%)
            min_holding_period: Minimum days to hold position (1)
            max_holding_period: Maximum days to hold position (10)
            rebalance_threshold: Rebalance when position drift exceeds threshold (5%)
        """

        # Initialize base strategy
        strategy_params = kwargs.copy()
        strategy_params.update(
            {
                "min_expected_return": return_threshold,
                "min_confidence": confidence_threshold,
                "max_positions": max_positions,
                "position_sizing": position_sizing,
            }
        )
        super().__init__(strategy_params)

        # ML-specific parameters
        self.return_threshold = return_threshold
        self.confidence_threshold = confidence_threshold
        self.correlation_threshold = correlation_threshold
        self.volatility_target = volatility_target
        self.max_position_size = max_position_size
        self.min_holding_period = min_holding_period
        self.max_holding_period = max_holding_period
        self.rebalance_threshold = rebalance_threshold

        # Position tracking
        self.position_entry_dates = {}
        self.position_correlations = {}
        self.position_expected_returns = {}

        logger.info(
            f"MLThresholdStrategy initialized: "
            f"return_threshold={return_threshold:.1%}, "
            f"confidence_threshold={confidence_threshold:.1%}, "
            f"max_positions={max_positions}"
        )

    def generate_signals(
        self, predictions: pd.DataFrame, positions: Dict, market_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """
        Generate trading signals from ML predictions

        Args:
            predictions: DataFrame with columns ['ticker', 'predicted_return_*', 'confidence']
                        or Series indexed by ticker with prediction values
            positions: Current portfolio positions
            market_data: Historical market data for correlation and volatility analysis

        Returns:
            List of validated trading signals
        """
        signals = []

        try:
            # Parse predictions into structured format
            ml_predictions = self._parse_predictions(predictions)

            if not ml_predictions:
                logger.debug("No predictions to process")
                return signals

            # Generate entry signals
            entry_signals = self._generate_entry_signals(ml_predictions, positions, market_data)
            signals.extend(entry_signals)

            # Generate exit signals for existing positions
            exit_signals = self._generate_exit_signals(positions, market_data, ml_predictions)
            signals.extend(exit_signals)

            # Generate rebalancing signals
            rebalance_signals = self._generate_rebalance_signals(positions, ml_predictions)
            signals.extend(rebalance_signals)

            logger.debug(
                f"Generated {len(signals)} signals: "
                f"{len(entry_signals)} entry, {len(exit_signals)} exit, "
                f"{len(rebalance_signals)} rebalance"
            )

        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")

        return signals

    def _parse_predictions(self, predictions: pd.DataFrame) -> List[MLPrediction]:
        """
        Parse raw predictions into structured MLPrediction objects

        Args:
            predictions: Raw prediction data

        Returns:
            List of structured MLPrediction objects
        """
        ml_predictions = []

        try:
            if isinstance(predictions, pd.Series):
                # Single day predictions indexed by ticker
                for ticker, pred_value in predictions.items():
                    if pd.isna(pred_value):
                        continue

                    # Handle different prediction formats
                    if isinstance(pred_value, dict):
                        ml_pred = MLPrediction(
                            ticker=ticker,
                            predicted_return_1d=pred_value.get("predicted_return_1d", 0.0),
                            predicted_return_3d=pred_value.get(
                                "predicted_return_3d", pred_value.get("return_3d", 0.0)
                            ),
                            predicted_return_5d=pred_value.get("predicted_return_5d", 0.0),
                            confidence=pred_value.get("confidence", 0.5),
                            volatility=pred_value.get("volatility", 0.02),
                            timestamp=datetime.now(),
                        )
                    else:
                        # Single value - assume it's 3-day return
                        ml_pred = MLPrediction(
                            ticker=ticker,
                            predicted_return_1d=float(pred_value) * 0.5,
                            predicted_return_3d=float(pred_value),
                            predicted_return_5d=float(pred_value) * 1.5,
                            confidence=0.6,  # Default confidence
                            volatility=0.02,  # Default volatility
                            timestamp=datetime.now(),
                        )

                    ml_predictions.append(ml_pred)

            elif isinstance(predictions, pd.DataFrame):
                # DataFrame with prediction data
                for idx, row in predictions.iterrows():
                    ticker = row.get("ticker", str(idx))

                    ml_pred = MLPrediction(
                        ticker=ticker,
                        predicted_return_1d=row.get("predicted_return_1d", 0.0),
                        predicted_return_3d=row.get(
                            "predicted_return_3d", row.get("return_3d", 0.0)
                        ),
                        predicted_return_5d=row.get("predicted_return_5d", 0.0),
                        confidence=row.get("confidence", 0.5),
                        volatility=row.get("volatility", 0.02),
                        timestamp=row.get("timestamp", datetime.now()),
                    )

                    ml_predictions.append(ml_pred)

        except Exception as e:
            logger.error(f"Error parsing predictions: {e}")

        return ml_predictions

    def _generate_entry_signals(
        self, ml_predictions: List[MLPrediction], positions: Dict, market_data: pd.DataFrame
    ) -> List[TradingSignal]:
        """Generate entry signals based on ML predictions"""
        entry_signals = []

        # Filter and rank predictions for entry
        entry_candidates = []

        for pred in ml_predictions:
            # Basic threshold checks
            if pred.primary_return < self.return_threshold:
                continue
            if pred.confidence < self.confidence_threshold:
                continue
            if pred.ticker in positions:
                continue  # Already have position

            entry_candidates.append(pred)

        if not entry_candidates:
            return entry_signals

        # Rank by risk-adjusted return
        entry_candidates.sort(key=lambda x: x.risk_adjusted_return, reverse=True)

        # Apply position limits and correlation constraints
        selected_candidates = self._apply_position_constraints(
            entry_candidates, positions, market_data
        )

        # Create signals with position sizing
        for pred in selected_candidates:
            try:
                position_size = self._calculate_ml_position_size(pred, positions, market_data)

                if position_size > 0:
                    signal = TradingSignal(
                        ticker=pred.ticker,
                        type="BUY",
                        shares=int(position_size * 100),  # Will be adjusted by risk manager
                        confidence=pred.confidence,
                        expected_return=pred.primary_return,
                        timestamp=pred.timestamp,
                        reason=f"ML Signal: Expected return {pred.primary_return:.2%}, "
                        f"Confidence {pred.confidence:.2%}, "
                        f"Risk-adjusted score {pred.risk_adjusted_return:.2%}",
                    )

                    entry_signals.append(signal)

                    # Track entry for position management
                    self.position_entry_dates[pred.ticker] = pred.timestamp
                    self.position_expected_returns[pred.ticker] = pred.primary_return

            except Exception as e:
                logger.error(f"Error creating entry signal for {pred.ticker}: {e}")

        return entry_signals

    def _generate_exit_signals(
        self, positions: Dict, market_data: pd.DataFrame, ml_predictions: List[MLPrediction]
    ) -> List[TradingSignal]:
        """Generate exit signals for existing positions"""
        exit_signals = []

        # Create prediction lookup
        pred_dict = {pred.ticker: pred for pred in ml_predictions}

        for ticker, position in positions.items():
            try:
                should_exit, exit_reason = self._should_exit_ml_position(
                    ticker, position, pred_dict.get(ticker), market_data
                )

                if should_exit:
                    signal = TradingSignal(
                        ticker=ticker,
                        type="SELL",
                        shares=position.shares,
                        confidence=0.9,  # High confidence for exits
                        expected_return=0.0,
                        timestamp=datetime.now(),
                        reason=exit_reason,
                    )

                    exit_signals.append(signal)

                    # Clean up tracking
                    if ticker in self.position_entry_dates:
                        del self.position_entry_dates[ticker]
                    if ticker in self.position_expected_returns:
                        del self.position_expected_returns[ticker]

            except Exception as e:
                logger.error(f"Error checking exit for {ticker}: {e}")

        return exit_signals

    def _generate_rebalance_signals(
        self, positions: Dict, ml_predictions: List[MLPrediction]
    ) -> List[TradingSignal]:
        """Generate rebalancing signals to adjust position sizes"""
        rebalance_signals = []

        try:
            # Calculate current position weights
            total_value = sum(pos.current_value for pos in positions.values())
            if total_value == 0:
                return rebalance_signals

            # Create prediction lookup
            pred_dict = {pred.ticker: pred for pred in ml_predictions}

            for ticker, position in positions.items():
                current_weight = position.current_value / total_value
                pred = pred_dict.get(ticker)

                if pred is None:
                    continue

                # Calculate target weight based on updated predictions
                target_weight = self._calculate_target_weight(pred, positions)
                weight_diff = target_weight - current_weight

                # Check if rebalancing is needed
                if abs(weight_diff) > self.rebalance_threshold:
                    # Calculate shares to adjust
                    if weight_diff > 0:
                        # Increase position
                        shares_to_buy = int(weight_diff * total_value / position.current_price)
                        if shares_to_buy > 0:
                            signal = TradingSignal(
                                ticker=ticker,
                                type="BUY",
                                shares=shares_to_buy,
                                confidence=pred.confidence,
                                expected_return=pred.primary_return,
                                timestamp=datetime.now(),
                                reason=f"Rebalance: Increase position by {weight_diff:.1%}",
                            )
                            rebalance_signals.append(signal)
                    else:
                        # Decrease position
                        shares_to_sell = int(
                            abs(weight_diff) * total_value / position.current_price
                        )
                        shares_to_sell = min(shares_to_sell, position.shares)
                        if shares_to_sell > 0:
                            signal = TradingSignal(
                                ticker=ticker,
                                type="SELL",
                                shares=shares_to_sell,
                                confidence=0.8,
                                expected_return=0.0,
                                timestamp=datetime.now(),
                                reason=f"Rebalance: Decrease position by {abs(weight_diff):.1%}",
                            )
                            rebalance_signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating rebalance signals: {e}")

        return rebalance_signals

    def _apply_position_constraints(
        self, candidates: List[MLPrediction], positions: Dict, market_data: pd.DataFrame
    ) -> List[MLPrediction]:
        """Apply position limits and correlation constraints"""
        selected = []

        # Check maximum positions limit
        available_slots = self.max_positions - len(positions)
        if available_slots <= 0:
            return selected

        # Calculate correlations for existing positions
        existing_tickers = list(positions.keys())

        for candidate in candidates[: available_slots * 2]:  # Consider more candidates
            # Check correlation with existing positions
            if self._check_correlation_constraint(candidate.ticker, existing_tickers, market_data):
                selected.append(candidate)
                existing_tickers.append(candidate.ticker)  # Update for next iteration

                if len(selected) >= available_slots:
                    break

        return selected

    def _check_correlation_constraint(
        self, new_ticker: str, existing_tickers: List[str], market_data: pd.DataFrame
    ) -> bool:
        """Check if new ticker violates correlation constraints"""
        if not existing_tickers:
            return True

        try:
            # Get return data for correlation calculation
            if isinstance(market_data.index, pd.MultiIndex):
                # Handle MultiIndex data
                returns_data = {}
                for ticker in existing_tickers + [new_ticker]:
                    ticker_data = market_data.loc[market_data.index.get_level_values(1) == ticker]
                    if len(ticker_data) > 20:  # Minimum data points
                        returns = ticker_data["Close"].pct_change().dropna()
                        returns_data[ticker] = returns
            else:
                # Handle regular index with ticker columns
                all_tickers = existing_tickers + [new_ticker]
                available_tickers = [t for t in all_tickers if t in market_data.columns]

                if len(available_tickers) < len(all_tickers):
                    return True  # Allow if correlation data not available

                returns_data = market_data[available_tickers].pct_change().dropna()

            if len(returns_data) < 2:
                return True  # Allow if insufficient data

            # Calculate correlations
            returns_df = pd.DataFrame(returns_data)
            if new_ticker not in returns_df.columns:
                return True  # Allow if new ticker data not available

            correlations = returns_df.corrwith(returns_df[new_ticker])

            # Check maximum correlation
            max_correlation = correlations[existing_tickers].max()
            return max_correlation < self.correlation_threshold

        except Exception as e:
            logger.warning(f"Error checking correlation for {new_ticker}: {e}")
            return True  # Allow if correlation check fails

    def _calculate_ml_position_size(
        self, prediction: MLPrediction, positions: Dict, market_data: pd.DataFrame
    ) -> float:
        """
        Calculate position size using ML-enhanced methods

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0)
        """
        if self.position_sizing == "equal_weight":
            return 1.0 / self.max_positions

        elif self.position_sizing == "kelly":
            # Enhanced Kelly criterion with ML confidence
            win_prob = prediction.confidence
            expected_return = prediction.primary_return

            # Estimate loss probability and amount
            loss_prob = 1 - win_prob
            expected_loss = prediction.volatility * 2  # Assume 2-sigma loss

            if expected_loss > 0:
                kelly_fraction = (
                    win_prob * expected_return - loss_prob * expected_loss
                ) / expected_return

                # Apply safety factor and ML confidence adjustment
                safety_factor = 0.25 * prediction.confidence  # Scale safety by confidence
                kelly_fraction = max(0, min(kelly_fraction * safety_factor, self.max_position_size))

                return kelly_fraction
            else:
                return self.max_position_size / self.max_positions

        elif self.position_sizing == "volatility_scaled":
            # Inverse volatility weighting with ML adjustment
            base_size = self.max_position_size / self.max_positions
            vol_adjustment = self.volatility_target / max(prediction.volatility, 0.005)
            confidence_adjustment = prediction.confidence

            position_size = base_size * vol_adjustment * confidence_adjustment
            return max(0.01, min(position_size, self.max_position_size))

        elif self.position_sizing == "risk_parity":
            # Risk parity with ML risk-adjusted returns
            risk_contribution = prediction.volatility
            if risk_contribution > 0:
                target_risk = self.volatility_target / self.max_positions
                position_size = (target_risk / risk_contribution) * prediction.confidence
                return max(0.01, min(position_size, self.max_position_size))
            else:
                return self.max_position_size / self.max_positions

        else:
            # Default fixed sizing
            return self.max_position_size / self.max_positions

    def _should_exit_ml_position(
        self, ticker: str, position, prediction: Optional[MLPrediction], market_data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Enhanced exit logic incorporating ML predictions

        Returns:
            (should_exit, exit_reason)
        """
        # Standard exit conditions
        should_exit, exit_reason = self._check_standard_exits(ticker, position)
        if should_exit:
            return True, exit_reason

        # ML-specific exit conditions
        if prediction:
            # Exit if prediction turns negative
            if prediction.primary_return < -0.005:  # -0.5% threshold
                return True, f"ML prediction turned negative: {prediction.primary_return:.2%}"

            # Exit if confidence drops significantly
            if prediction.confidence < 0.3:
                return True, f"ML confidence too low: {prediction.confidence:.2%}"

            # Exit if risk-adjusted return becomes poor
            if prediction.risk_adjusted_return < 0:
                return True, f"Poor risk-adjusted return: {prediction.risk_adjusted_return:.2%}"

        # Time-based exits with ML adjustment
        if ticker in self.position_entry_dates:
            days_held = (datetime.now() - self.position_entry_dates[ticker]).days

            # Dynamic holding period based on prediction quality
            if prediction:
                max_hold = self.max_holding_period * prediction.confidence
            else:
                max_hold = self.max_holding_period * 0.5  # Conservative if no prediction

            if days_held >= max_hold:
                return True, f"Maximum holding period reached: {days_held} days"

        return False, ""

    def _check_standard_exits(self, ticker: str, position) -> Tuple[bool, str]:
        """Check standard exit conditions"""
        # Stop loss
        if hasattr(position, "unrealized_pnl_pct"):
            if position.unrealized_pnl_pct <= -self.stop_loss:
                return True, f"Stop loss triggered: {position.unrealized_pnl_pct:.2%}"

        # Profit target
        if hasattr(position, "unrealized_pnl_pct"):
            if position.unrealized_pnl_pct >= self.profit_target:
                return True, f"Profit target reached: {position.unrealized_pnl_pct:.2%}"

        # Minimum holding period
        if ticker in self.position_entry_dates:
            days_held = (datetime.now() - self.position_entry_dates[ticker]).days
            if days_held < self.min_holding_period:
                return False, ""  # Too early to exit

        return False, ""

    def _calculate_target_weight(self, prediction: MLPrediction, positions: Dict) -> float:
        """Calculate target weight for position based on updated ML prediction"""
        # Use ML-enhanced position sizing
        base_weight = self._calculate_ml_position_size(prediction, positions, pd.DataFrame())

        # Adjust based on prediction quality
        quality_factor = prediction.confidence * (1 + max(0, prediction.primary_return))
        adjusted_weight = base_weight * quality_factor

        return max(0.01, min(adjusted_weight, self.max_position_size))

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get comprehensive strategy state including ML parameters"""
        base_state = super().get_strategy_state()

        ml_state = {
            "return_threshold": self.return_threshold,
            "confidence_threshold": self.confidence_threshold,
            "correlation_threshold": self.correlation_threshold,
            "volatility_target": self.volatility_target,
            "max_position_size": self.max_position_size,
            "min_holding_period": self.min_holding_period,
            "max_holding_period": self.max_holding_period,
            "rebalance_threshold": self.rebalance_threshold,
            "current_positions": len(self.position_entry_dates),
            "tracked_positions": list(self.position_entry_dates.keys()),
        }

        base_state.update(ml_state)
        return base_state
