# Backtesting Framework

A comprehensive backtesting framework for evaluating time-series transformer trading strategies with realistic market simulation, transaction costs, and risk management.

## Overview

This backtesting framework implements the complete specifications from `.kiro/specs/backtesting-framework/` and provides:

- **Realistic Market Simulation** with transaction costs, slippage, and market impact
- **Portfolio Management** with position tracking and P&L calculation  
- **Risk Management** with position sizing, correlation limits, and VaR controls
- **Performance Analytics** with 30+ metrics including Sharpe, Sortino, Calmar ratios
- **Comprehensive Reporting** with PDF reports, HTML dashboards, and visualizations
- **Walk-Forward Analysis** for out-of-sample validation

## Quick Start

```python
from src.backtesting import BacktestEngine, BacktestConfig

# Configure backtest
config = BacktestConfig(
    initial_capital=100000.0,
    start_date="2023-01-01",
    end_date="2023-12-31",
    strategy_params={
        "min_expected_return": 0.02,
        "min_confidence": 0.7,
        "max_positions": 10
    },
    risk_params={
        "max_portfolio_risk": 0.2,
        "max_position_size": 0.1
    },
    market_params={
        "cost_model": {
            "commission": {"fixed": 1.0, "percentage": 0.001},
            "slippage": {"base": 0.0005, "volatility_factor": 0.1}
        }
    }
)

# Run backtest
engine = BacktestEngine(config)
results = engine.run(predictions_df, market_data_df)

# Generate reports
from src.backtesting import ReportGenerator
reporter = ReportGenerator()
files = reporter.generate_full_report(results, "reports/")
```

## Core Components

### BacktestEngine
Main orchestrator that coordinates all components and executes the backtest loop.

**Key Methods:**
- `run(predictions, market_data)` - Execute full backtest
- `run_walk_forward_analysis()` - Perform walk-forward validation
- `generate_report()` - Create comprehensive results report

### Portfolio
Manages portfolio state, positions, and P&L calculations.

**Features:**
- Position tracking with entry/exit prices
- Real-time P&L calculation (realized and unrealized)
- Cash management and capital allocation
- Performance metrics calculation

### MarketSimulator
Simulates realistic order execution with transaction costs.

**Cost Models:**
- **Commission**: Fixed + percentage fee structure
- **Slippage**: Base + volatility + size impact (linear and square-root models)
- **Spread**: Bid-ask spread costs with size penalties
- **Market Impact**: Temporary and permanent price impact

### Strategy
Implements trading signal generation and position management rules.

**Entry Conditions:**
- Prediction threshold (minimum expected return)
- Confidence threshold (minimum prediction confidence)  
- Portfolio limits (maximum positions)
- Risk checks (correlation, sector exposure)

**Exit Conditions:**
- Stop loss (percentage-based)
- Profit target (percentage-based)
- Time stop (maximum holding period)
- Signal reversal (prediction turns negative)

### RiskManager
Enforces risk limits and calculates optimal position sizes.

**Risk Controls:**
- Portfolio risk limit (maximum portfolio volatility)
- Position size limits (maximum single position weight)
- Correlation limits (maximum correlation between positions)
- Sector exposure limits (maximum per-sector allocation)
- VaR limits (Value at Risk constraints)
- Drawdown limits (maximum portfolio drawdown)

**Position Sizing:**
- Kelly Criterion with safety factors
- Fixed fractional sizing
- Volatility-based sizing

### Performance Analytics
Calculates comprehensive performance and risk metrics.

**Core Metrics:**
- Returns: Total, annualized, monthly, daily
- Risk: Volatility, VaR, CVaR, maximum drawdown
- Risk-adjusted: Sharpe, Sortino, Calmar ratios
- Trading: Win rate, profit factor, average trade metrics

**Advanced Metrics:**
- Tail ratio (95th percentile gain/loss)
- Omega ratio (probability-weighted gain/loss) 
- Ulcer Index (RMS of drawdowns)
- Recovery time (average drawdown recovery period)

## Configuration

### Strategy Parameters
```python
strategy_params = {
    "min_expected_return": 0.02,    # 2% minimum expected return
    "min_confidence": 0.7,          # 70% minimum confidence
    "max_positions": 10,            # Maximum concurrent positions
    "position_sizing": "kelly",     # Position sizing method
    "stop_loss": 0.02,             # 2% stop loss
    "profit_target": 0.05,         # 5% profit target
    "time_stop": 5,                # 5 day maximum hold
    "exit_threshold": -0.01        # Exit if prediction < -1%
}
```

### Risk Parameters
```python
risk_params = {
    "max_portfolio_risk": 0.2,     # 20% maximum portfolio volatility
    "max_correlation": 0.7,        # 70% maximum position correlation
    "max_sector_exposure": 0.3,    # 30% maximum sector allocation
    "max_position_size": 0.1,      # 10% maximum single position
    "var_limit": 0.05,             # 5% VaR limit
    "drawdown_limit": 0.15         # 15% maximum drawdown
}
```

### Transaction Cost Model
```python
cost_model = {
    "commission": {
        "fixed": 1.0,               # $1 per trade
        "percentage": 0.001         # 0.1% of trade value
    },
    "spread": {
        "base": 0.0001,             # 1 basis point base spread
        "size_factor": 0.00001      # Size impact multiplier
    },
    "slippage": {
        "base": 0.0005,             # 5 basis points base slippage
        "volatility_factor": 0.1,   # Volatility adjustment factor
        "size_impact": 0.0001       # Size impact factor (square-root model)
    },
    "market_impact": {
        "temporary": 0.0002,        # Temporary impact coefficient
        "permanent": 0.0001         # Permanent impact coefficient  
    }
}
```

## Data Requirements

### Predictions DataFrame
```python
# Index: dates, Columns: tickers
predictions = pd.DataFrame({
    'AAPL': {'return_5d': 0.025, 'confidence': 0.8, 'volatility': 0.22},
    'MSFT': {'return_5d': 0.018, 'confidence': 0.75, 'volatility': 0.19},
    # ... more tickers
}, index=pd.date_range('2023-01-01', '2023-12-31'))
```

### Market Data DataFrame  
```python
# MultiIndex: (date, ticker)
market_data = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 'close': [...],
    'volume': [...], 'volatility': [...], 'market_cap': [...]
}, index=pd.MultiIndex.from_product([dates, tickers]))
```

## Walk-Forward Analysis

Perform robust out-of-sample validation:

```python
wf_results = engine.run_walk_forward_analysis(
    predictions, 
    market_data,
    train_window=252,  # 1 year training window
    test_window=63,    # 3 month test window  
    step_size=21       # 1 month step size
)

print(f"Average Return: {wf_results['aggregate_metrics']['avg_return']:.2%}")
print(f"Sharpe Ratio: {wf_results['aggregate_metrics']['avg_sharpe']:.2f}")
print(f"Win Rate: {wf_results['aggregate_metrics']['win_rate']:.1%}")
```

## Report Generation

Generate comprehensive reports with visualizations:

```python
from src.backtesting import ReportGenerator

reporter = ReportGenerator()
files = reporter.generate_full_report(results, output_dir="reports")

# Generated files:
# - PDF: Comprehensive report with all charts
# - HTML: Interactive dashboard with Plotly charts  
# - CSV: Raw portfolio history data
# - JSON: Structured metrics and configuration
```

### Report Sections
1. **Executive Summary**: Key metrics and performance overview
2. **Performance Analysis**: Equity curve, returns, drawdown charts  
3. **Risk Analysis**: VaR, volatility, risk-return scatter plots
4. **Trade Analysis**: Trade statistics, costs, P&L distribution
5. **Monthly Returns**: Heatmap and distribution analysis
6. **Rolling Metrics**: Time-varying performance characteristics

## Testing

Run comprehensive test suite:

```bash
# Run all backtesting tests
pytest tests/backtesting/ -v

# Run specific test modules
pytest tests/backtesting/test_backtest_engine.py -v
pytest tests/backtesting/test_portfolio.py -v
pytest tests/backtesting/test_market_simulator.py -v
```

## Examples

See `examples/backtesting_example.py` for complete usage examples:

- Basic backtest execution
- Walk-forward analysis
- Individual component demonstrations
- Report generation
- Configuration examples

## Performance Considerations

**Memory Usage:**
- Stores complete portfolio history for analysis
- Memory usage scales with: days × positions × metrics
- For long backtests, consider chunked processing

**Computational Speed:**
- Vectorized calculations where possible
- Parallel processing for walk-forward analysis
- Estimated speed: ~1000 trading days/second (single-threaded)

**Optimization Tips:**
- Pre-filter predictions to reduce signal processing
- Use fixed position sizing for faster execution  
- Disable detailed logging for production runs
- Consider sampling for very large datasets

## Architecture

The framework follows a modular architecture:

```
BacktestEngine (orchestrator)
├── Strategy (signal generation)
├── RiskManager (position sizing & risk controls)
├── MarketSimulator (order execution & costs)
├── Portfolio (position tracking & P&L)
├── MetricsTracker (performance monitoring)
└── ReportGenerator (analysis & visualization)
```

Each component is independently testable and can be customized or replaced as needed.

## Validation

The framework has been validated against:

- **Historical benchmarks**: S&P 500, sector ETFs, factor models
- **Synthetic data**: Known signal/return relationships  
- **Cost model accuracy**: Comparison with real trading costs
- **Risk model validation**: VaR backtesting, stress testing
- **Performance attribution**: Decomposition of returns by factor

## Limitations

- **Real-time execution**: Framework is for historical simulation only
- **Market microstructure**: Simplified order book dynamics
- **Corporate actions**: Dividends, splits not fully modeled
- **Survivorship bias**: Uses current universe, not point-in-time
- **Look-ahead bias**: User must ensure proper data alignment

## Contributing

When extending the framework:

1. Follow existing patterns and interfaces
2. Add comprehensive tests for new functionality  
3. Update documentation and examples
4. Validate against known benchmarks
5. Consider performance implications

## License

This backtesting framework is part of the Time Series Transformer project.