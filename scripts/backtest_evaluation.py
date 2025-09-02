"""
Comprehensive Backtesting Framework
Tests model predictions against multiple baselines with realistic trading constraints
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.timeseries_transformer import TimeSeriesTransformer


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% per trade
        self.slippage = slippage  # 0.05% slippage
        self.reset()
    
    def reset(self):
        """Reset strategy to initial state"""
        self.capital = self.initial_capital
        self.positions = {}  # {ticker: shares}
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        self.dates = []
    
    def execute_trade(self, ticker, shares, price, date):
        """Execute a trade with transaction costs"""
        # Ensure shares is a scalar value
        if hasattr(shares, 'item'):
            shares = shares.item()
        elif hasattr(shares, '__len__'):
            shares = float(shares)
        
        if shares == 0:
            return
        
        # Apply slippage (worse price for us)
        if shares > 0:  # Buying
            execution_price = price * (1 + self.slippage)
        else:  # Selling
            execution_price = price * (1 - self.slippage)
        
        # Calculate trade value
        trade_value = shares * execution_price
        transaction_fee = abs(trade_value) * self.transaction_cost
        
        # Update capital
        self.capital -= trade_value + transaction_fee
        
        # Update positions
        if ticker not in self.positions:
            self.positions[ticker] = 0
        self.positions[ticker] += shares
        
        # Record trade
        self.trades.append({
            'date': date,
            'ticker': ticker,
            'shares': shares,
            'price': execution_price,
            'value': trade_value,
            'fee': transaction_fee,
            'capital_after': self.capital
        })
        
        return execution_price
    
    def calculate_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        total = self.capital
        for ticker, shares in self.positions.items():
            if ticker in prices:
                total += shares * prices[ticker]
        return total


class ModelBacktester:
    """Backtester for the trained model predictions"""
    
    def __init__(self, model_path='model_extended_best.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = TimeSeriesTransformer(
            input_dim=21,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            forecast_horizon=3,
            output_dim=3
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def prepare_features_simple(self, df):
        """Simplified feature preparation matching training"""
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # Volatility
        df['volatility_20d'] = df['returns'].rolling(20).std()
        
        # Bollinger Bands
        bb_mean = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_position'] = (df['Close'] - bb_mean) / (2 * bb_std + 1e-8)
        
        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Market regime
        df['trend_strength'] = (df['MA_50'] - df['MA_200']) / (df['MA_200'] + 1e-8)
        df['distance_from_MA200'] = (df['Close'] - df['MA_200']) / (df['MA_200'] + 1e-8)
        
        # Temporal
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter
        
        return df
    
    def backtest(self, tickers, start_date, end_date, rebalance_freq='3D'):
        """Run backtest on historical data"""
        
        print(f"\nBacktesting from {start_date} to {end_date}")
        print(f"Rebalancing every: {rebalance_freq}")
        print("="*60)
        
        # Calculate extended start date for feature calculation
        # Need at least 250 days before start_date for MA_200 and other features
        extended_start = pd.to_datetime(start_date) - pd.DateOffset(days=365)
        
        # Download historical data with extended period
        data = {}
        print(f"Downloading data from {extended_start.date()} (includes feature warmup period)")
        
        for ticker in tickers:
            df = yf.download(ticker, start=extended_start, end=end_date, progress=False)
            if len(df) > 250:  # Need enough history for features
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} days of data")
            else:
                print(f"  ✗ {ticker}: Insufficient data ({len(df)} days)")
        
        if len(data) == 0:
            print("No sufficient data available")
            return None
        
        print(f"\nUsing {len(data)} stocks for backtesting")
        
        # Initialize strategies
        model_strategy = TradingStrategy()
        buy_hold_strategy = TradingStrategy()
        random_strategy = TradingStrategy()
        
        # Find first valid trading day after start_date
        actual_start_date = pd.to_datetime(start_date)
        first_valid_date = None
        
        for ticker in data:
            ticker_dates = data[ticker].index[data[ticker].index >= actual_start_date]
            if len(ticker_dates) > 0:
                if first_valid_date is None or ticker_dates[0] < first_valid_date:
                    first_valid_date = ticker_dates[0]
        
        if first_valid_date is None:
            print("No valid trading dates found")
            return None
        
        print(f"Actual backtest start: {first_valid_date.date()}")
        
        # Buy and hold initialization
        initial_prices = {}
        for ticker in data:
            if first_valid_date in data[ticker].index:
                initial_price = float(data[ticker].loc[first_valid_date, 'Close'])
                initial_prices[ticker] = initial_price
                shares = buy_hold_strategy.initial_capital / len(data) / initial_price
                buy_hold_strategy.execute_trade(ticker, shares, initial_price, 
                                               first_valid_date)
        
        # Prepare for backtesting loop
        trading_days = pd.date_range(start=first_valid_date, 
                                     end=end_date, freq='B')
        
        rebalance_dates = pd.date_range(start=trading_days[0], 
                                        end=trading_days[-1], 
                                        freq=rebalance_freq)
        
        print(f"Trading days: {len(trading_days)}")
        print(f"Rebalance dates: {len(rebalance_dates)}")
        
        # Backtesting loop
        print("\nRunning backtest simulation...")
        progress_counter = 0
        
        for date in trading_days:
            progress_counter += 1
            if progress_counter % 50 == 0:
                print(f"  Progress: {progress_counter}/{len(trading_days)} days...")
            
            current_prices = {}
            
            # Get current prices
            for ticker in data:
                ticker_data = data[ticker]
                if date in ticker_data.index:
                    current_prices[ticker] = float(ticker_data.loc[date, 'Close'])
            
            if len(current_prices) == 0:
                continue
            
            # Record portfolio values
            model_strategy.portfolio_values.append(
                model_strategy.calculate_portfolio_value(current_prices)
            )
            model_strategy.dates.append(date)
            
            buy_hold_strategy.portfolio_values.append(
                buy_hold_strategy.calculate_portfolio_value(current_prices)
            )
            
            random_strategy.portfolio_values.append(
                random_strategy.calculate_portfolio_value(current_prices)
            )
            
            # Rebalance if needed
            if date in rebalance_dates:
                # Model predictions
                predictions = self.generate_predictions(data, date)
                
                if predictions:
                    # Rank by predicted returns
                    ranked = sorted(predictions.items(), 
                                  key=lambda x: x[1], reverse=True)
                    
                    # Clear existing positions
                    for ticker in list(model_strategy.positions.keys()):
                        if ticker in current_prices:
                            shares = model_strategy.positions[ticker]
                            if shares != 0:
                                model_strategy.execute_trade(
                                    ticker, -shares, 
                                    current_prices[ticker], date
                                )
                    
                    # Take new positions (top 3 long, bottom 3 short)
                    position_size = model_strategy.capital / 6
                    
                    # Long positions
                    for ticker, pred_return in ranked[:3]:
                        if ticker in current_prices and pred_return > 0:
                            shares = position_size / current_prices[ticker]
                            model_strategy.execute_trade(
                                ticker, shares, 
                                current_prices[ticker], date
                            )
                    
                    # Short positions (if predictions are negative)
                    for ticker, pred_return in ranked[-3:]:
                        if ticker in current_prices and pred_return < 0:
                            shares = -position_size / current_prices[ticker]
                            model_strategy.execute_trade(
                                ticker, shares, 
                                current_prices[ticker], date
                            )
                
                # Random strategy rebalancing
                for ticker in list(random_strategy.positions.keys()):
                    if ticker in current_prices:
                        shares = random_strategy.positions[ticker]
                        if shares != 0:
                            random_strategy.execute_trade(
                                ticker, -shares, 
                                current_prices[ticker], date
                            )
                
                # Random positions
                random_tickers = np.random.choice(list(current_prices.keys()), 
                                                 size=min(3, len(current_prices)), 
                                                 replace=False)
                position_size = random_strategy.capital / len(random_tickers)
                
                for ticker in random_tickers:
                    shares = position_size / current_prices[ticker]
                    if np.random.random() > 0.5:  # Random long/short
                        shares = -shares
                    random_strategy.execute_trade(
                        ticker, shares, 
                        current_prices[ticker], date
                    )
        
        # Calculate metrics
        results = {
            'model': self.calculate_metrics(model_strategy),
            'buy_hold': self.calculate_metrics(buy_hold_strategy),
            'random': self.calculate_metrics(random_strategy)
        }
        
        # Add strategy objects for plotting
        results['strategies'] = {
            'model': model_strategy,
            'buy_hold': buy_hold_strategy,
            'random': random_strategy
        }
        
        return results
    
    def generate_predictions(self, data, current_date):
        """Generate model predictions for current date"""
        predictions = {}
        
        for ticker, ticker_data in data.items():
            try:
                # Get data up to current date
                available_data = ticker_data[ticker_data.index <= current_date]
                
                if len(available_data) < 250:
                    continue
                
                # Prepare features
                df_features = self.prepare_features_simple(available_data.copy())
                
                # Select feature columns (matching training)
                feature_cols = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'returns', 'returns', 'volatility_20d',
                    'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                    'BB_position', 'volume_ratio',
                    'MA_50', 'MA_200', 'trend_strength', 'distance_from_MA200',
                    'day_of_week', 'month', 'quarter'
                ]
                
                # Get last 60 days
                features = df_features[feature_cols].iloc[-60:].values
                
                # Handle NaN
                features = np.nan_to_num(features, 0)
                
                # Standardize
                features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
                
                # Predict
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred = self.model(x).cpu().numpy()[0]
                
                # Average prediction for 3 days
                predictions[ticker] = float(np.mean(pred))
                
            except Exception as e:
                continue
        
        return predictions
    
    def calculate_metrics(self, strategy):
        """Calculate performance metrics"""
        returns = pd.Series(strategy.portfolio_values).pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (strategy.portfolio_values[-1] / strategy.portfolio_values[0] - 1)
        
        # Annualized metrics
        n_days = len(strategy.portfolio_values)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Risk metrics
        daily_returns = returns
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
        
        # Drawdown
        cumulative = pd.Series(strategy.portfolio_values) / strategy.portfolio_values[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        n_trades = len(strategy.trades)
        
        if n_trades > 0:
            trade_returns = []
            for trade in strategy.trades:
                if trade['shares'] < 0:  # Closing position
                    # Simple approximation of trade return
                    trade_returns.append(trade['value'] / abs(trade['value']))
            
            if trade_returns:
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'final_value': strategy.portfolio_values[-1]
        }
    
    def plot_results(self, results, save_path='backtest_results.png'):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Portfolio values over time
        for name, strategy in results['strategies'].items():
            if len(strategy.dates) > 0:
                axes[0, 0].plot(strategy.dates, strategy.portfolio_values[1:], 
                               label=name.replace('_', ' ').title())
        
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns comparison
        strategy_names = []
        returns = []
        
        for name in ['model', 'buy_hold', 'random']:
            if name in results and 'total_return' in results[name]:
                strategy_names.append(name.replace('_', ' ').title())
                returns.append(results[name]['total_return'] * 100)
        
        colors = ['red' if r < 0 else 'green' for r in returns]
        axes[0, 1].bar(strategy_names, returns, color=colors)
        axes[0, 1].set_title('Total Returns Comparison')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe ratios
        sharpes = []
        for name in ['model', 'buy_hold', 'random']:
            if name in results and 'sharpe_ratio' in results[name]:
                sharpes.append(results[name]['sharpe_ratio'])
            else:
                sharpes.append(0)
        
        axes[0, 2].bar(strategy_names, sharpes, color=['blue', 'green', 'orange'])
        axes[0, 2].set_title('Sharpe Ratio Comparison')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Drawdown chart
        for name, strategy in results['strategies'].items():
            if len(strategy.portfolio_values) > 1:
                cumulative = pd.Series(strategy.portfolio_values) / strategy.portfolio_values[0]
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                axes[1, 0].fill_between(range(len(drawdown)), 0, drawdown * 100,
                                       alpha=0.3, label=name.replace('_', ' ').title())
        
        axes[1, 0].set_title('Drawdown Over Time')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Trade distribution
        model_trades = results['strategies']['model'].trades
        if model_trades:
            trade_values = [t['value'] for t in model_trades]
            axes[1, 1].hist(trade_values, bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title(f'Model Trade Distribution (n={len(model_trades)})')
            axes[1, 1].set_xlabel('Trade Value ($)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(x=0, color='red', linestyle='--')
        else:
            axes[1, 1].text(0.5, 0.5, 'No trades executed', 
                           ha='center', va='center')
        
        # Summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        # Create summary data
        summary_data = []
        for name in ['model', 'buy_hold', 'random']:
            if name in results:
                row = [
                    name.replace('_', ' ').title(),
                    f"{results[name].get('total_return', 0)*100:.2f}%",
                    f"{results[name].get('sharpe_ratio', 0):.2f}",
                    f"{results[name].get('max_drawdown', 0)*100:.2f}%",
                    f"{results[name].get('n_trades', 0)}"
                ]
                summary_data.append(row)
        
        table = axes[1, 2].table(cellText=summary_data,
                                colLabels=['Strategy', 'Return', 'Sharpe', 'Max DD', 'Trades'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.suptitle('Backtest Results Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nResults saved to {save_path}")


def main():
    """Run comprehensive backtest"""
    
    print("="*60)
    print("COMPREHENSIVE BACKTESTING FRAMEWORK")
    print("="*60)
    
    # Configuration - Using 2024 data for more recent backtesting
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'JPM']
    START_DATE = '2024-01-01'  # Start of backtest period
    END_DATE = '2024-08-01'     # End of backtest period (8 months)
    
    print(f"\nConfiguration:")
    print(f"  Stocks: {', '.join(TICKERS)}")
    print(f"  Backtest Period: {START_DATE} to {END_DATE}")
    print(f"  Initial Capital: $100,000")
    print(f"  Transaction Cost: 0.1%")
    print(f"  Slippage: 0.05%")
    
    # Initialize backtester
    try:
        backtester = ModelBacktester()
    except FileNotFoundError:
        print("\nError: Model file not found.")
        print("Please ensure 'model_extended_best.pt' exists in the current directory.")
        return
    
    # Run backtest
    results = backtester.backtest(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        rebalance_freq='3D'  # Rebalance every 3 days
    )
    
    if results is None:
        print("Backtesting failed - insufficient data")
        return
    
    # Print detailed results
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    for strategy_name in ['model', 'buy_hold', 'random']:
        if strategy_name in results:
            print(f"\n{strategy_name.replace('_', ' ').upper()} STRATEGY:")
            print("-" * 40)
            
            metrics = results[strategy_name]
            print(f"Total Return: {metrics.get('total_return', 0)*100:+.2f}%")
            print(f"Annual Return: {metrics.get('annual_return', 0)*100:+.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"Number of Trades: {metrics.get('n_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"Final Portfolio Value: ${metrics.get('final_value', 0):,.2f}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    model_return = results['model'].get('total_return', 0)
    bh_return = results['buy_hold'].get('total_return', 0)
    random_return = results['random'].get('total_return', 0)
    
    print(f"\nModel vs Buy & Hold: {(model_return - bh_return)*100:+.2f}%")
    print(f"Model vs Random: {(model_return - random_return)*100:+.2f}%")
    
    if model_return > bh_return:
        print("✅ Model OUTPERFORMED Buy & Hold")
    else:
        print("❌ Model UNDERPERFORMED Buy & Hold")
    
    if model_return > random_return:
        print("✅ Model OUTPERFORMED Random Strategy")
    else:
        print("❌ Model UNDERPERFORMED Random Strategy")
    
    # Risk-adjusted performance
    model_sharpe = results['model'].get('sharpe_ratio', 0)
    bh_sharpe = results['buy_hold'].get('sharpe_ratio', 0)
    
    print(f"\nRisk-Adjusted Performance (Sharpe):")
    print(f"Model: {model_sharpe:.2f}")
    print(f"Buy & Hold: {bh_sharpe:.2f}")
    
    if model_sharpe > bh_sharpe:
        print("✅ Better risk-adjusted returns than Buy & Hold")
    else:
        print("❌ Worse risk-adjusted returns than Buy & Hold")
    
    # Generate plots
    backtester.plot_results(results)
    
    # Save detailed results
    save_results = {
        'configuration': {
            'tickers': TICKERS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': 100000
        },
        'metrics': {
            'model': results['model'],
            'buy_hold': results['buy_hold'],
            'random': results['random']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print("\nDetailed results saved to backtest_results.json")
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if model_return > 0 and model_return > bh_return:
        print("The model shows promise but needs improvement.")
    elif model_return > 0:
        print("The model is profitable but doesn't beat passive investing.")
    else:
        print("The model loses money and needs fundamental redesign.")
        print("\nLessons learned:")
        print("1. Model predictions lack sufficient differentiation")
        print("2. Transaction costs significantly impact performance")
        print("3. More diverse features and longer training needed")
        print("4. Consider simpler models or different approaches")


if __name__ == "__main__":
    main()