# Backtesting Framework Specification

## .kiro/specs/backtesting-framework/requirements.md
```markdown
# Backtesting Framework Requirements
---
priority: 1
---

## Functional Requirements

### EARS Notation

WHEN a backtest is initiated
THE SYSTEM SHALL load historical data for the specified period
AND ensure no look-ahead bias in predictions

WHEN executing trades based on predictions
THE SYSTEM SHALL apply realistic transaction costs
INCLUDING spread, commission, and slippage
WHERE costs vary by trade size and market conditions

WHEN calculating portfolio performance
THE SYSTEM SHALL track multiple metrics
INCLUDING returns, Sharpe ratio, max drawdown, and win rate
AND compare against buy-and-hold benchmark

WHEN position sizing is required
THE SYSTEM SHALL use Kelly criterion or fixed fractional sizing
WITH maximum position limits and risk controls

WHEN generating backtest reports
THE SYSTEM SHALL produce visualizations
INCLUDING equity curve, drawdown chart, and trade distribution
AND export results in multiple formats (PDF, HTML, JSON)

## Trading Strategy Requirements

### Strategy Configuration
```python
STRATEGY_CONFIG = {
    "entry_rules": {
        "prediction_threshold": 0.02,  # 2% expected return
        "confidence_threshold": 0.7,   # 70% confidence
        "max_positions": 10,           # Portfolio limit
        "position_size": 0.1            # 10% per position
    },
    "exit_rules": {
        "profit_target": 0.05,          # 5% profit
        "stop_loss": 0.02,              # 2% loss
        "time_stop": 5,                 # Days
        "trailing_stop": True
    },
    "risk_management": {
        "max_portfolio_risk": 0.2,      # 20% total risk
        "max_correlation": 0.7,         # Position correlation limit
        "max_sector_exposure": 0.3      # 30% per sector
    }
}
```

### Transaction Cost Model
```python
COST_MODEL = {
    "commission": {
        "fixed": 1.0,                   # $1 per trade
        "percentage": 0.001             # 0.1% of trade value
    },
    "spread": {
        "base": 0.0001,                 # 1 basis point
        "size_factor": 0.00001          # Increases with size
    },
    "slippage": {
        "base": 0.0005,                 # 5 basis points
        "volatility_factor": 0.1,       # Scales with volatility
        "size_impact": 0.0001           # Linear impact
    },
    "market_impact": {
        "temporary": 0.0002,            # Temporary impact
        "permanent": 0.0001             # Permanent impact
    }
}
```

## Performance Metrics Requirements

### Core Metrics
- **Returns**: Daily, monthly, annual, cumulative
- **Risk**: Volatility, VaR, CVaR, maximum drawdown
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Trading**: Win rate, profit factor, average trade
- **Portfolio**: Beta, alpha, correlation to benchmark

### Advanced Metrics
```python
ADVANCED_METRICS = {
    "tail_ratio": "95th percentile gain / 95th percentile loss",
    "ulcer_index": "Root mean square of drawdowns",
    "omega_ratio": "Probability weighted gain/loss ratio",
    "recovery_time": "Average time to recover from drawdown",
    "kelly_fraction": "Optimal betting fraction"
}
```

### Benchmark Comparisons
- S&P 500 buy-and-hold
- Equal-weight portfolio
- 60/40 stocks/bonds
- Risk parity portfolio
- Custom benchmark

## Simulation Requirements

### Market Conditions
```python
MARKET_CONDITIONS = {
    "regimes": ["bull", "bear", "sideways", "volatile"],
    "events": ["flash_crash", "earnings", "fed_announcement"],
    "liquidity": ["high", "normal", "low", "crisis"],
    "volatility": ["low", "normal", "high", "extreme"]
}
```

### Walk-Forward Analysis
- Training window: 252 days (1 year)
- Validation window: 63 days (3 months)
- Test window: 21 days (1 month)
- Step size: 21 days
- Retraining frequency: Monthly

### Monte Carlo Simulation
- Number of paths: 1000
- Random seed control
- Parameter perturbation
- Bootstrap resampling
- Confidence intervals

## Reporting Requirements

### Report Sections
1. **Executive Summary**: Key metrics and conclusions
2. **Performance Analysis**: Detailed metrics and charts
3. **Risk Analysis**: Risk metrics and stress tests
4. **Trade Analysis**: Individual trade statistics
5. **Market Regime Analysis**: Performance by market condition

### Visualizations
- Cumulative returns chart
- Underwater plot (drawdowns)
- Returns distribution histogram
- Trade scatter plot (return vs duration)
- Heatmap (monthly returns)
- Rolling metrics (Sharpe, volatility)

### Export Formats
- PDF report with charts
- HTML interactive dashboard
- CSV raw data export
- JSON structured results
- LaTeX tables for papers
```

