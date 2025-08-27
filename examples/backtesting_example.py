"""
Complete example demonstrating the backtesting framework usage

This example shows how to:
1. Set up a backtest configuration
2. Prepare prediction and market data
3. Run a backtest
4. Generate reports
5. Perform walk-forward analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backtesting import (
    BacktestEngine, BacktestConfig, Portfolio, MarketSimulator,
    Strategy, RiskManager, MetricsTracker, PerformanceAnalyzer, ReportGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample prediction and market data for testing"""
    
    # Date range for backtest
    start_date = "2023-01-01"
    end_date = "2023-06-30"
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    logger.info(f"Creating sample data for {len(dates)} days and {len(tickers)} tickers")
    
    # Create sample predictions
    predictions_data = []
    for date in dates:
        for ticker in tickers:
            # Simulate model predictions with some realistic characteristics
            base_return = np.random.normal(0.001, 0.02)  # Small positive bias
            confidence = np.random.beta(7, 3)  # Skewed toward higher confidence
            volatility = np.random.uniform(0.15, 0.4)
            
            predictions_data.append({
                'date': date,
                'ticker': ticker,
                'return_5d': base_return,
                'confidence': confidence,
                'volatility': volatility
            })
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions = predictions_df.pivot(index='date', columns='ticker', 
                                     values=['return_5d', 'confidence', 'volatility'])
    
    # Create sample market data
    market_data_list = []
    base_prices = {"AAPL": 150, "MSFT": 250, "GOOGL": 100, "TSLA": 200, "NVDA": 300}
    
    for date in dates:
        for ticker in tickers:
            # Random walk with drift
            if market_data_list:
                prev_price = [x for x in market_data_list if x['ticker'] == ticker][-1]['close']
            else:
                prev_price = base_prices[ticker]
            
            daily_return = np.random.normal(0.0005, 0.025)  # Slight positive drift
            price = prev_price * (1 + daily_return)
            
            market_data_list.append({
                'date': date,
                'ticker': ticker,
                'open': price * np.random.uniform(0.995, 1.005),
                'high': price * np.random.uniform(1.002, 1.015),
                'low': price * np.random.uniform(0.985, 0.998),
                'close': price,
                'volume': np.random.randint(1000000, 10000000),
                'volatility': predictions_df[
                    (predictions_df['date'] == date) & 
                    (predictions_df['ticker'] == ticker)
                ]['volatility'].iloc[0] if len(predictions_df[
                    (predictions_df['date'] == date) & 
                    (predictions_df['ticker'] == ticker)
                ]) > 0 else 0.2,
                'market_cap': np.random.uniform(100e9, 2000e9),  # 100B to 2T
                'sector': np.random.choice(['Technology', 'Consumer', 'Industrial'])
            })
    
    market_data = pd.DataFrame(market_data_list).set_index(['date', 'ticker'])
    
    logger.info(f"Sample data created: {len(predictions)} prediction days, {len(market_data)} market data points")
    
    return predictions, market_data


def create_backtest_configuration():
    """Create comprehensive backtest configuration"""
    
    config = BacktestConfig(
        initial_capital=100000.0,  # $100k starting capital
        start_date="2023-01-01",
        end_date="2023-06-30",
        
        # Strategy parameters
        strategy_params={
            "min_expected_return": 0.015,      # 1.5% minimum expected return
            "min_confidence": 0.65,            # 65% minimum confidence
            "max_positions": 8,                # Maximum 8 concurrent positions
            "position_sizing": "kelly",        # Use Kelly criterion
            "stop_loss": 0.03,                 # 3% stop loss
            "profit_target": 0.08,             # 8% profit target
            "time_stop": 10,                   # 10 day time stop
            "exit_threshold": -0.005           # Exit when prediction turns negative
        },
        
        # Risk management parameters
        risk_params={
            "max_portfolio_risk": 0.15,        # 15% max portfolio risk
            "max_correlation": 0.6,            # 60% max correlation between positions
            "max_sector_exposure": 0.4,        # 40% max exposure per sector
            "max_position_size": 0.12,         # 12% max single position size
            "var_limit": 0.04,                 # 4% VaR limit
            "concentration_limit": 0.2,        # 20% concentration limit
            "drawdown_limit": 0.12,            # 12% max drawdown
            "lookback_days": 252               # 1 year lookback for risk calculations
        },
        
        # Market simulation parameters
        market_params={
            "cost_model": {
                "commission": {
                    "fixed": 1.0,               # $1 fixed commission
                    "percentage": 0.0005        # 0.05% of trade value
                },
                "spread": {
                    "base": 0.0001,             # 1 basis point base spread
                    "size_factor": 0.00001      # Size impact on spread
                },
                "slippage": {
                    "base": 0.0003,             # 3 basis points base slippage
                    "volatility_factor": 0.08,  # Volatility impact factor
                    "size_impact": 0.00008      # Size impact factor
                },
                "market_impact": {
                    "temporary": 0.0001,        # Temporary impact
                    "permanent": 0.00005        # Permanent impact
                }
            },
            "execution_delay": 0,               # No execution delay
            "market_hours": {"start": "09:30", "end": "16:00"}
        }
    )
    
    logger.info("Backtest configuration created")
    return config


def run_basic_backtest_example():
    """Run a basic backtest example"""
    
    logger.info("=== Starting Basic Backtest Example ===")
    
    # Create sample data and configuration
    predictions, market_data = create_sample_data()
    config = create_backtest_configuration()
    
    # Initialize backtest engine
    engine = BacktestEngine(config)
    
    logger.info("Running backtest...")
    
    # Run the backtest
    # For this example, we'll use just the return_5d predictions
    predictions_simple = predictions['return_5d'].fillna(0)
    
    try:
        results = engine.run(predictions_simple, market_data)
        
        # Display results
        logger.info("=== BACKTEST RESULTS ===")
        metrics = results['metrics']
        
        print(f"\nBACKTEST SUMMARY")
        print(f"================")
        print(f"Period: {config.start_date} to {config.end_date}")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Final Value: ${metrics.get('final_value', 0):,.2f}")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Total Trades: {metrics.get('total_trades', 0):,}")
        
        # Generate reports
        logger.info("Generating reports...")
        report_generator = ReportGenerator()
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate full report
        generated_files = report_generator.generate_full_report(results, str(reports_dir))
        
        print(f"\nREPORTS GENERATED:")
        for report_type, file_path in generated_files.items():
            print(f"  {report_type.upper()}: {file_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


def run_walk_forward_example():
    """Run walk-forward analysis example"""
    
    logger.info("=== Starting Walk-Forward Analysis Example ===")
    
    # Create extended dataset for walk-forward
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    dates = pd.date_range(start_date, end_date, freq='D')
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Create predictions
    predictions_data = []
    for date in dates:
        for ticker in tickers:
            predictions_data.append({
                'date': date,
                'ticker': ticker,
                'return_5d': np.random.normal(0.001, 0.02),
                'confidence': np.random.beta(6, 4),
                'volatility': np.random.uniform(0.15, 0.4)
            })
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions = predictions_df.pivot(index='date', columns='ticker', values='return_5d').fillna(0)
    
    # Create market data
    market_data_list = []
    base_prices = {"AAPL": 150, "MSFT": 250, "GOOGL": 100}
    
    for date in dates:
        for ticker in tickers:
            price = base_prices[ticker] * (1 + np.random.normal(0, 0.3))  # Random walk
            
            market_data_list.append({
                'date': date,
                'ticker': ticker,
                'close': price,
                'volume': np.random.randint(1000000, 5000000),
                'volatility': np.random.uniform(0.15, 0.35)
            })
    
    market_data = pd.DataFrame(market_data_list).set_index(['date', 'ticker'])
    
    # Configuration for walk-forward
    config = BacktestConfig(
        initial_capital=50000.0,
        start_date=start_date,
        end_date=end_date,
        strategy_params={"min_expected_return": 0.01},
        risk_params={"max_portfolio_risk": 0.2},
        market_params={"cost_model": {}}
    )
    
    engine = BacktestEngine(config)
    
    logger.info("Running walk-forward analysis...")
    
    try:
        wf_results = engine.run_walk_forward_analysis(
            predictions, 
            market_data,
            train_window=252,  # 1 year training
            test_window=63,    # 3 months testing
            step_size=21       # 1 month step
        )
        
        # Display walk-forward results
        print(f"\nWALK-FORWARD ANALYSIS RESULTS")
        print(f"=============================")
        agg_metrics = wf_results['aggregate_metrics']
        print(f"Number of Periods: {agg_metrics['num_periods']}")
        print(f"Average Return: {agg_metrics['avg_return']:.2%}")
        print(f"Return Std Dev: {agg_metrics['std_return']:.2%}")
        print(f"Average Sharpe: {agg_metrics['avg_sharpe']:.2f}")
        print(f"Win Rate: {agg_metrics['win_rate']:.1%}")
        print(f"Best Period: {agg_metrics['best_period']:.2%}")
        print(f"Worst Period: {agg_metrics['worst_period']:.2%}")
        
        return wf_results
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        raise


def demonstrate_individual_components():
    """Demonstrate individual component functionality"""
    
    logger.info("=== Demonstrating Individual Components ===")
    
    # 1. Portfolio Management
    print("\n1. Portfolio Management Example:")
    portfolio = Portfolio(initial_capital=50000)
    print(f"   Initial capital: ${portfolio.total_value:,.2f}")
    print(f"   Cash: ${portfolio.cash:,.2f}")
    print(f"   Positions: {len(portfolio.positions)}")
    
    # 2. Risk Management
    print("\n2. Risk Manager Example:")
    risk_config = {
        "max_portfolio_risk": 0.15,
        "max_position_size": 0.1,
        "var_limit": 0.05
    }
    risk_manager = RiskManager(risk_config)
    print(f"   Max portfolio risk: {risk_manager.max_portfolio_risk:.1%}")
    print(f"   Max position size: {risk_manager.max_position_size:.1%}")
    
    # 3. Performance Analytics
    print("\n3. Performance Analytics Example:")
    analyzer = PerformanceAnalyzer(risk_free_rate=0.03)
    
    # Simulate some returns
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    sample_values = [10000 * (1 + sample_returns.iloc[:i+1]).prod() for i in range(len(sample_returns))]
    
    # Calculate metrics
    portfolio_history = [{"portfolio_value": v, "date": datetime.now()} for v in sample_values]
    metrics = analyzer.calculate_metrics(portfolio_history)
    
    print(f"   Sample Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Sample Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   Sample Total Return: {metrics.get('total_return', 0):.2%}")
    
    # 4. Market Simulation
    print("\n4. Market Simulator Example:")
    market_config = {
        "cost_model": {
            "commission": {"fixed": 1.0, "percentage": 0.001},
            "slippage": {"base": 0.0005}
        }
    }
    market_sim = MarketSimulator(market_config)
    print(f"   Commission model: ${market_config['cost_model']['commission']['fixed']} + {market_config['cost_model']['commission']['percentage']:.1%}")
    print(f"   Base slippage: {market_config['cost_model']['slippage']['base']:.2%}")


def main():
    """Main execution function"""
    
    print("Time Series Transformer - Backtesting Framework Example")
    print("=" * 60)
    
    try:
        # 1. Run basic backtest
        basic_results = run_basic_backtest_example()
        
        # 2. Demonstrate individual components
        demonstrate_individual_components()
        
        # 3. Run walk-forward analysis (commented out to save time)
        # wf_results = run_walk_forward_example()
        
        print(f"\n=== EXAMPLE COMPLETED SUCCESSFULLY ===")
        print(f"Check the 'reports' directory for generated files.")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()