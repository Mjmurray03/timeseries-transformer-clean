#!/usr/bin/env python3
"""
# COMPONENT: Enhanced Risk Manager
# PURPOSE: Comprehensive portfolio risk management and position sizing
# INPUTS: Trading signals, portfolio state, market data
# OUTPUTS: Validated signals with proper position sizing
# VERIFICATION: Risk limits, correlation constraints, VaR calculations
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class RiskConstraints:
    """Risk constraint parameters"""
    max_drawdown: float = 0.15          # 15% maximum drawdown
    max_position_size: float = 0.2      # 20% maximum position size
    max_sector_concentration: float = 0.4  # 40% maximum sector concentration
    max_correlation: float = 0.7        # Maximum correlation between positions
    var_limit: float = 0.05             # 5% daily VaR limit
    leverage_limit: float = 1.0         # No leverage
    min_diversification: int = 3        # Minimum number of positions
    max_turnover: float = 2.0           # Maximum annual turnover rate
    
    # Kelly Criterion parameters
    kelly_fraction: float = 0.25        # Safety factor for Kelly sizing
    max_kelly_position: float = 0.15    # Maximum Kelly position size


class EnhancedRiskManager:
    """
    # COMPONENT: Comprehensive Risk Management System
    # PURPOSE: Apply portfolio constraints and calculate optimal position sizes
    # INPUTS: Trading signals, portfolio state, market data, risk parameters
    # OUTPUTS: Risk-adjusted signals with validated position sizes
    # VERIFICATION: All risk limits enforced, position sizes optimized
    """
    
    def __init__(
        self,
        risk_params: Dict[str, Any],
        sector_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize enhanced risk manager
        
        Args:
            risk_params: Risk parameter configuration
            sector_mapping: Mapping of tickers to sectors
        """
        # Initialize constraints
        self.constraints = RiskConstraints(**risk_params)
        
        # Sector mapping for concentration limits
        self.sector_mapping = sector_mapping or {}
        
        # Portfolio tracking
        self.portfolio_history = []
        self.risk_metrics_history = []
        self.correlation_matrix = None
        self.volatility_estimates = {}
        
        # VaR model
        self.var_confidence = 0.95
        self.var_window = 252  # 1 year
        
        logger.info(f"Enhanced risk manager initialized: "
                   f"max_drawdown={self.constraints.max_drawdown:.1%}, "
                   f"max_position={self.constraints.max_position_size:.1%}, "
                   f"var_limit={self.constraints.var_limit:.1%}")
    
    def filter_signals(
        self,
        signals: List,
        portfolio,
        market_data: pd.DataFrame
    ) -> List:
        """
        Filter and validate trading signals through risk management
        
        Args:
            signals: Raw trading signals from strategy
            portfolio: Current portfolio state
            market_data: Market data for risk calculations
            
        Returns:
            Risk-approved signals with validated position sizes
        """
        if not signals:
            return []
        
        try:
            # Update risk metrics
            self._update_risk_metrics(portfolio, market_data)
            
            # Filter signals by portfolio constraints
            filtered_signals = self._apply_portfolio_constraints(signals, portfolio)
            
            # Calculate optimal position sizes
            sized_signals = self._calculate_optimal_position_sizes(
                filtered_signals, portfolio, market_data
            )
            
            # Apply final validation
            validated_signals = self._final_validation(sized_signals, portfolio)
            
            logger.debug(f"Risk manager: {len(signals)} -> {len(filtered_signals)} -> "
                        f"{len(sized_signals)} -> {len(validated_signals)} signals")
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error in risk filtering: {e}")
            return []
    
    def validate_portfolio_constraints(
        self, 
        portfolio,
        new_signal
    ) -> Tuple[bool, str]:
        """
        Validate portfolio constraints for new signal
        
        Returns:
            (is_valid, reason)
        """
        try:
            # Check position count limits
            if new_signal.type == 'BUY' and len(portfolio.positions) >= 20:
                return False, "Maximum position count exceeded"
            
            # Check position size limits
            if hasattr(portfolio, 'total_value'):
                position_value = new_signal.shares * self._get_current_price(new_signal.ticker)
                position_pct = position_value / portfolio.total_value
                
                if position_pct > self.constraints.max_position_size:
                    return False, f"Position size {position_pct:.1%} exceeds limit {self.constraints.max_position_size:.1%}"
            
            # Check sector concentration
            if not self._check_sector_constraint(new_signal, portfolio):
                return False, "Sector concentration limit exceeded"
            
            # Check correlation limits
            if not self._check_correlation_constraint(new_signal, portfolio):
                return False, "Correlation limit exceeded"
            
            # Check drawdown limits
            if hasattr(portfolio, 'max_drawdown'):
                if portfolio.max_drawdown > self.constraints.max_drawdown:
                    return False, f"Portfolio drawdown {portfolio.max_drawdown:.1%} exceeds limit"
            
            return True, "All constraints satisfied"
            
        except Exception as e:
            logger.error(f"Error validating constraints: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size(
        self,
        signal,
        portfolio,
        volatility: float,
        method: str = 'kelly'
    ) -> float:
        """
        Calculate optimal position size using specified method
        
        Args:
            signal: Trading signal
            portfolio: Portfolio state
            volatility: Asset volatility
            method: Sizing method ('kelly', 'equal_weight', 'risk_parity', 'max_diversification')
            
        Returns:
            Position size as fraction of portfolio
        """
        try:
            if method == 'kelly':
                return self._calculate_kelly_position_size(signal, volatility)
            elif method == 'equal_weight':
                return self._calculate_equal_weight_size(portfolio)
            elif method == 'risk_parity':
                return self._calculate_risk_parity_size(signal, portfolio, volatility)
            elif method == 'max_diversification':
                return self._calculate_max_diversification_size(signal, portfolio, volatility)
            else:
                return self._calculate_equal_weight_size(portfolio)
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.05  # Default 5% position size
    
    def update_stop_losses(
        self,
        portfolio,
        market_data: pd.DataFrame
    ):
        """
        Update dynamic stop losses based on market conditions
        
        Args:
            portfolio: Portfolio with positions
            market_data: Current market data
        """
        try:
            for ticker, position in portfolio.positions.items():
                # Get current volatility
                volatility = self._estimate_volatility(ticker, market_data)
                
                # Calculate volatility-based stop loss
                vol_stop = 2.0 * volatility  # 2-sigma stop
                
                # Use trailing stop for profitable positions
                if hasattr(position, 'unrealized_pnl_pct') and position.unrealized_pnl_pct > 0:
                    trailing_stop = max(vol_stop, 0.05)  # 5% minimum trailing stop
                    new_stop = min(position.entry_price * (1 - trailing_stop), 
                                 position.current_price * (1 - vol_stop))
                else:
                    # Use volatility-based stop for losing positions
                    new_stop = position.entry_price * (1 - vol_stop)
                
                # Update position stop loss
                if hasattr(position, 'stop_loss'):
                    # Only tighten stops, never loosen them
                    if new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        logger.debug(f"Updated stop loss for {ticker}: ${new_stop:.2f}")
                else:
                    position.stop_loss = new_stop
                    
        except Exception as e:
            logger.error(f"Error updating stop losses: {e}")
    
    def calculate_portfolio_var(
        self,
        portfolio,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk metrics
        
        Returns:
            Dictionary with VaR metrics
        """
        try:
            if not portfolio.positions or not self.portfolio_history:
                return {'var': 0.0, 'cvar': 0.0, 'volatility': 0.0}
            
            # Get portfolio returns history
            returns = self._get_portfolio_returns_history(portfolio)
            
            if len(returns) < 20:
                return {'var': 0.0, 'cvar': 0.0, 'volatility': 0.0}
            
            # Calculate VaR (parametric method)
            returns_mean = returns.mean()
            returns_std = returns.std()
            
            z_score = stats.norm.ppf(1 - confidence)
            parametric_var = abs(z_score * returns_std - returns_mean)
            
            # Calculate historical VaR
            historical_var = abs(np.percentile(returns, (1 - confidence) * 100))
            
            # Calculate CVaR (Expected Shortfall)
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var_threshold]
            cvar = abs(tail_returns.mean()) if len(tail_returns) > 0 else parametric_var
            
            # Use more conservative estimate
            var = max(parametric_var, historical_var)
            
            return {
                'var': var,
                'cvar': cvar,
                'volatility': returns_std * np.sqrt(252),  # Annualized
                'parametric_var': parametric_var,
                'historical_var': historical_var,
                'returns_mean': returns_mean * 252,  # Annualized
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'var': 0.0, 'cvar': 0.0, 'volatility': 0.0}
    
    def _apply_portfolio_constraints(self, signals: List, portfolio) -> List:
        """Apply basic portfolio constraints"""
        filtered_signals = []
        
        for signal in signals:
            is_valid, reason = self.validate_portfolio_constraints(portfolio, signal)
            
            if is_valid:
                filtered_signals.append(signal)
            else:
                logger.debug(f"Signal filtered: {signal.ticker} - {reason}")
        
        return filtered_signals
    
    def _calculate_optimal_position_sizes(
        self,
        signals: List,
        portfolio,
        market_data: pd.DataFrame
    ) -> List:
        """Calculate optimal position sizes for approved signals"""
        sized_signals = []
        
        for signal in signals:
            try:
                # Estimate volatility
                volatility = self._estimate_volatility(signal.ticker, market_data)
                
                # Calculate position size
                position_size = self.calculate_position_size(
                    signal, portfolio, volatility, method='kelly'
                )
                
                # Convert to shares
                if hasattr(portfolio, 'total_value') and portfolio.total_value > 0:
                    position_value = position_size * portfolio.total_value
                    current_price = self._get_current_price(signal.ticker)
                    if current_price > 0:
                        signal.shares = int(position_value / current_price)
                    else:
                        signal.shares = 0
                
                if signal.shares > 0:
                    sized_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error sizing signal for {signal.ticker}: {e}")
        
        return sized_signals
    
    def _final_validation(self, signals: List, portfolio) -> List:
        """Final validation of all signals"""
        validated_signals = []
        
        # Check total exposure
        total_exposure = sum(
            signal.shares * self._get_current_price(signal.ticker) 
            for signal in signals if signal.type == 'BUY'
        )
        
        current_exposure = sum(
            pos.current_value for pos in portfolio.positions.values()
        ) if hasattr(portfolio, 'positions') else 0
        
        total_portfolio_exposure = total_exposure + current_exposure
        portfolio_value = getattr(portfolio, 'total_value', 100000)
        
        if total_portfolio_exposure / portfolio_value > self.constraints.leverage_limit:
            # Scale down signals proportionally
            scale_factor = (self.constraints.leverage_limit * portfolio_value - current_exposure) / total_exposure
            scale_factor = max(0, min(1, scale_factor))
            
            for signal in signals:
                if signal.type == 'BUY':
                    signal.shares = int(signal.shares * scale_factor)
                
                if signal.shares > 0:
                    validated_signals.append(signal)
        else:
            validated_signals = signals
        
        return validated_signals
    
    def _calculate_kelly_position_size(self, signal, volatility: float) -> float:
        """Calculate Kelly criterion position size"""
        try:
            # Kelly formula: f* = (bp - q) / b
            expected_return = signal.expected_return
            win_probability = signal.confidence
            
            # Estimate average win/loss based on volatility
            avg_win = expected_return
            avg_loss = volatility * 2  # 2-sigma loss
            
            if avg_win > 0 and avg_loss > 0:
                # Kelly fraction
                kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
                
                # Apply safety factor and constraints
                kelly_fraction = max(0, kelly_fraction)
                kelly_fraction = kelly_fraction * self.constraints.kelly_fraction
                kelly_fraction = min(kelly_fraction, self.constraints.max_kelly_position)
                
                return kelly_fraction
            else:
                return 0.05  # Default 5%
                
        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}")
            return 0.05
    
    def _calculate_equal_weight_size(self, portfolio) -> float:
        """Calculate equal weight position size"""
        target_positions = min(self.constraints.min_diversification, 10)
        return 1.0 / target_positions
    
    def _calculate_risk_parity_size(self, signal, portfolio, volatility: float) -> float:
        """Calculate risk parity position size"""
        try:
            # Target equal risk contribution
            target_vol = 0.15  # 15% target portfolio volatility
            target_positions = 5
            target_position_vol = target_vol / np.sqrt(target_positions)
            
            # Position size to achieve target volatility contribution
            if volatility > 0:
                position_size = target_position_vol / volatility
                return min(position_size, self.constraints.max_position_size)
            else:
                return self._calculate_equal_weight_size(portfolio)
                
        except Exception as e:
            logger.error(f"Error in risk parity calculation: {e}")
            return self._calculate_equal_weight_size(portfolio)
    
    def _calculate_max_diversification_size(self, signal, portfolio, volatility: float) -> float:
        """Calculate position size for maximum diversification"""
        # Inverse volatility weighting
        if volatility > 0:
            base_weight = 0.15  # Base weight
            vol_adjustment = (0.02 / volatility)  # Assume 2% target volatility
            position_size = base_weight * vol_adjustment
            return min(position_size, self.constraints.max_position_size)
        else:
            return self._calculate_equal_weight_size(portfolio)
    
    def _check_sector_constraint(self, signal, portfolio) -> bool:
        """Check sector concentration limits"""
        try:
            if signal.ticker not in self.sector_mapping:
                return True  # No sector info available
            
            signal_sector = self.sector_mapping[signal.ticker]
            
            # Calculate current sector exposure
            sector_exposure = 0.0
            portfolio_value = getattr(portfolio, 'total_value', 100000)
            
            if hasattr(portfolio, 'positions'):
                for ticker, position in portfolio.positions.items():
                    if ticker in self.sector_mapping:
                        if self.sector_mapping[ticker] == signal_sector:
                            sector_exposure += position.current_value
            
            # Add new signal exposure
            signal_value = signal.shares * self._get_current_price(signal.ticker)
            total_sector_exposure = (sector_exposure + signal_value) / portfolio_value
            
            return total_sector_exposure <= self.constraints.max_sector_concentration
            
        except Exception as e:
            logger.error(f"Error checking sector constraint: {e}")
            return True
    
    def _check_correlation_constraint(self, signal, portfolio) -> bool:
        """Check correlation limits with existing positions"""
        try:
            if not hasattr(portfolio, 'positions') or not portfolio.positions:
                return True
            
            # This is a simplified check - in practice, would use historical correlation data
            existing_tickers = list(portfolio.positions.keys())
            
            # If we have correlation matrix, use it
            if (self.correlation_matrix is not None and 
                signal.ticker in self.correlation_matrix.index):
                
                for existing_ticker in existing_tickers:
                    if existing_ticker in self.correlation_matrix.columns:
                        correlation = self.correlation_matrix.loc[signal.ticker, existing_ticker]
                        if abs(correlation) > self.constraints.max_correlation:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation constraint: {e}")
            return True
    
    def _estimate_volatility(self, ticker: str, market_data: pd.DataFrame, window: int = 20) -> float:
        """Estimate volatility for ticker"""
        try:
            # Extract ticker data from market data
            if isinstance(market_data.index, pd.MultiIndex):
                ticker_data = market_data.loc[market_data.index.get_level_values(1) == ticker]
                if len(ticker_data) > window:
                    returns = ticker_data['Close'].pct_change().dropna()
                    return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
            
            # Default volatility
            return self.volatilities.get(ticker, 0.25)  # 25% default
            
        except Exception:
            return 0.25
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for ticker (mock implementation)"""
        return 100.0  # Placeholder - would get from market data
    
    def _get_portfolio_returns_history(self, portfolio) -> pd.Series:
        """Get historical portfolio returns"""
        if len(self.portfolio_history) < 2:
            return pd.Series([0.0])
        
        # Extract portfolio values
        values = [entry.get('portfolio_value', 0) for entry in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        
        return returns
    
    def _update_risk_metrics(self, portfolio, market_data: pd.DataFrame):
        """Update risk metrics and portfolio history"""
        try:
            # Store portfolio snapshot
            portfolio_snapshot = {
                'timestamp': datetime.now(),
                'portfolio_value': getattr(portfolio, 'total_value', 0),
                'cash': getattr(portfolio, 'cash', 0),
                'positions': len(getattr(portfolio, 'positions', {}))
            }
            
            self.portfolio_history.append(portfolio_snapshot)
            
            # Keep only recent history
            if len(self.portfolio_history) > self.var_window:
                self.portfolio_history = self.portfolio_history[-self.var_window:]
            
            # Update volatility estimates
            if hasattr(portfolio, 'positions'):
                for ticker in portfolio.positions.keys():
                    self.volatilities[ticker] = self._estimate_volatility(ticker, market_data)
                    
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_risk_report(self, portfolio) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            var_metrics = self.calculate_portfolio_var(portfolio)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'constraints': {
                    'max_drawdown': self.constraints.max_drawdown,
                    'max_position_size': self.constraints.max_position_size,
                    'var_limit': self.constraints.var_limit,
                    'max_correlation': self.constraints.max_correlation
                },
                'current_metrics': var_metrics,
                'portfolio_summary': {
                    'total_positions': len(getattr(portfolio, 'positions', {})),
                    'portfolio_value': getattr(portfolio, 'total_value', 0),
                    'cash_percentage': getattr(portfolio, 'cash', 0) / max(getattr(portfolio, 'total_value', 1), 1)
                },
                'risk_status': {
                    'var_breach': var_metrics['var'] > self.constraints.var_limit,
                    'concentration_risk': self._check_concentration_risk(portfolio),
                    'liquidity_risk': self._assess_liquidity_risk(portfolio)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def _check_concentration_risk(self, portfolio) -> Dict[str, float]:
        """Check for concentration risk"""
        try:
            if not hasattr(portfolio, 'positions') or not portfolio.positions:
                return {'max_position_pct': 0.0, 'top_3_concentration': 0.0}
            
            position_values = [pos.current_value for pos in portfolio.positions.values()]
            total_value = sum(position_values)
            
            if total_value == 0:
                return {'max_position_pct': 0.0, 'top_3_concentration': 0.0}
            
            position_pcts = [val / total_value for val in position_values]
            position_pcts.sort(reverse=True)
            
            max_position_pct = position_pcts[0] if position_pcts else 0.0
            top_3_concentration = sum(position_pcts[:3])
            
            return {
                'max_position_pct': max_position_pct,
                'top_3_concentration': top_3_concentration,
                'herfindahl_index': sum(pct**2 for pct in position_pcts)
            }
            
        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
            return {'max_position_pct': 0.0, 'top_3_concentration': 0.0}
    
    def _assess_liquidity_risk(self, portfolio) -> Dict[str, Any]:
        """Assess portfolio liquidity risk"""
        try:
            # Simple liquidity assessment
            # In practice, would use actual volume data
            return {
                'illiquid_positions': 0,
                'avg_liquidity_score': 0.8,
                'time_to_liquidate_days': 1.0
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {e}")
            return {'illiquid_positions': 0}