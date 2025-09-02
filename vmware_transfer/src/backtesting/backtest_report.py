"""
Comprehensive Backtesting Report Generation

Generates professional quant-style tearsheets with performance metrics,
risk analysis, and visualizations for backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class BacktestResults:
    """Container for backtesting results"""
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    benchmark_returns: pd.Series
    drawdowns: pd.Series
    rolling_sharpe: pd.Series
    rolling_volatility: pd.Series
    strategy_params: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    initial_capital: float


class BacktestReport:
    """
    Comprehensive backtesting report generator
    
    Creates professional quant-style tearsheets with:
    - Performance metrics (returns, Sharpe, drawdowns)
    - Risk analysis (VaR, volatility, beta)
    - Trading statistics (win rate, profit factor)
    - Detailed visualizations
    """
    
    def __init__(self, results: BacktestResults):
        """Initialize with backtest results"""
        self.results = results
        self.trading_days = 252
        
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report
        
        Args:
            save_path: Optional path to save report files
            
        Returns:
            Dictionary containing all computed metrics
        """
        print("Generating comprehensive backtesting report...")
        
        # Calculate all metrics
        performance_metrics = self._calculate_performance_metrics()
        risk_metrics = self._calculate_risk_metrics()
        trading_metrics = self._calculate_trading_metrics()
        
        # Create visualizations
        if save_path:
            self._create_tearsheet(save_path)
        
        # Combine all metrics
        report = {
            'performance': performance_metrics,
            'risk': risk_metrics,
            'trading': trading_metrics,
            'summary': self._create_executive_summary(performance_metrics, risk_metrics, trading_metrics)
        }
        
        # Print key results
        self._print_summary(report)
        
        return report
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        returns = self.results.returns
        portfolio_values = self.results.portfolio_values
        benchmark_returns = self.results.benchmark_returns
        
        # Basic return metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(self.trading_days)
        downside_vol = returns[returns < 0].std() * np.sqrt(self.trading_days)
        
        # Risk-adjusted returns
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown metrics
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean()
        
        # Benchmark comparison
        if len(benchmark_returns) > 0:
            benchmark_total = (1 + benchmark_returns).cumprod().iloc[-1] - 1
            benchmark_vol = benchmark_returns.std() * np.sqrt(self.trading_days)
            benchmark_sharpe = (benchmark_total * self.trading_days / len(benchmark_returns)) / benchmark_vol
            
            # Beta calculation
            covariance = np.cov(returns.dropna(), benchmark_returns.dropna())[0][1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            alpha = annualized_return - (beta * benchmark_total * self.trading_days / len(benchmark_returns))
            information_ratio = (annualized_return - benchmark_total * self.trading_days / len(benchmark_returns)) / (returns - benchmark_returns).std() / np.sqrt(self.trading_days)
        else:
            benchmark_total = 0
            benchmark_sharpe = 0
            beta = 0
            alpha = 0
            information_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'benchmark_return': benchmark_total,
            'benchmark_sharpe': benchmark_sharpe,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'downside_volatility': downside_vol
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns = self.results.returns
        portfolio_values = self.results.portfolio_values
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Drawdown duration analysis
        drawdowns = self.results.drawdowns
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def _calculate_trading_metrics(self) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        trades = self.results.trades
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade_return': 0,
                'avg_winning_trade': 0,
                'avg_losing_trade': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'trades_per_month': 0
            }
        
        # Basic trade statistics
        total_trades = len(trades)
        
        # Assume trades have 'pnl' or 'return' column
        if 'pnl' in trades.columns:
            trade_returns = trades['pnl']
        elif 'return' in trades.columns:
            trade_returns = trades['return']
        else:
            # Fallback: calculate from entry/exit prices if available
            if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
                trade_returns = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
            else:
                return {'error': 'No trade return data available'}
        
        # Win/Loss analysis
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average trade metrics
        avg_trade_return = trade_returns.mean()
        avg_winning_trade = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_losing_trade = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Best/Worst trades
        largest_win = trade_returns.max()
        largest_loss = trade_returns.min()
        
        # Trading frequency
        days_trading = (self.results.end_date - self.results.start_date).days
        trades_per_month = total_trades * 30 / days_trading if days_trading > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = consecutive_losses = 0
        max_consecutive_wins = max_consecutive_losses = 0
        
        for ret in trade_returns:
            if ret > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif ret < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_wins = consecutive_losses = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'trades_per_month': trades_per_month,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _create_tearsheet(self, save_path: str):
        """Create comprehensive visual tearsheet"""
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(4, 2, 1)
        self.results.portfolio_values.plot(ax=ax1, linewidth=2, label='Strategy')
        if len(self.results.benchmark_returns) > 0:
            benchmark_values = (1 + self.results.benchmark_returns).cumprod() * self.results.initial_capital
            benchmark_values.plot(ax=ax1, linewidth=2, label='Benchmark', alpha=0.7)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown Analysis
        ax2 = plt.subplot(4, 2, 2)
        self.results.drawdowns.plot(ax=ax2, kind='area', color='red', alpha=0.3)
        ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = plt.subplot(4, 2, 3)
        if hasattr(self.results, 'rolling_sharpe') and len(self.results.rolling_sharpe) > 0:
            self.results.rolling_sharpe.plot(ax=ax3, linewidth=2, color='green')
        ax3.set_title('Rolling Sharpe Ratio (60-day)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Good Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = plt.subplot(4, 2, 4)
        monthly_returns = self.results.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        if len(monthly_returns_pivot) > 0:
            sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1%', cmap='RdYlBu_r', 
                       center=0, ax=ax4, cbar_kws={'label': 'Monthly Return'})
        ax4.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Year')
        
        # 5. Return Distribution
        ax5 = plt.subplot(4, 2, 5)
        self.results.returns.hist(bins=50, ax=ax5, alpha=0.7, density=True, color='skyblue')
        self.results.returns.plot(kind='kde', ax=ax5, color='red', linewidth=2)
        ax5.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Daily Returns')
        ax5.set_ylabel('Density')
        ax5.axvline(x=self.results.returns.mean(), color='green', linestyle='--', label=f'Mean: {self.results.returns.mean():.3f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling Volatility
        ax6 = plt.subplot(4, 2, 6)
        if hasattr(self.results, 'rolling_volatility') and len(self.results.rolling_volatility) > 0:
            (self.results.rolling_volatility * np.sqrt(252)).plot(ax=ax6, linewidth=2, color='orange')
        ax6.set_title('Rolling Volatility (60-day)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Annualized Volatility')
        ax6.grid(True, alpha=0.3)
        
        # 7. Position Exposure Over Time
        ax7 = plt.subplot(4, 2, 7)
        if len(self.results.positions) > 0 and 'total_exposure' in self.results.positions.columns:
            self.results.positions['total_exposure'].plot(ax=ax7, linewidth=2, color='purple')
        ax7.set_title('Portfolio Exposure Over Time', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Total Exposure ($)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Trade Analysis
        ax8 = plt.subplot(4, 2, 8)
        if len(self.results.trades) > 0:
            trade_returns = self.results.trades.get('pnl', self.results.trades.get('return', pd.Series()))
            if len(trade_returns) > 0:
                colors = ['green' if x > 0 else 'red' for x in trade_returns]
                ax8.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7)
                ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax8.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Trade Number')
        ax8.set_ylabel('Trade Return')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/backtest_tearsheet.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tearsheet saved to {save_path}/backtest_tearsheet.png")
    
    def _create_executive_summary(self, performance: Dict, risk: Dict, trading: Dict) -> Dict[str, Any]:
        """Create executive summary of key metrics"""
        return {
            'total_return': performance.get('total_return', 0),
            'annualized_return': performance.get('annualized_return', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown': performance.get('max_drawdown', 0),
            'win_rate': trading.get('win_rate', 0),
            'total_trades': trading.get('total_trades', 0),
            'var_95': risk.get('var_95', 0),
            'calmar_ratio': performance.get('calmar_ratio', 0)
        }
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print formatted summary to console"""
        print("\n" + "="*80)
        print("                        BACKTESTING RESULTS SUMMARY")
        print("="*80)
        
        perf = report['performance']
        risk = report['risk']
        trading = report['trading']
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"   Total Return:        {perf['total_return']:>8.2%}")
        print(f"   Annualized Return:   {perf['annualized_return']:>8.2%}")
        print(f"   Volatility:          {perf['volatility']:>8.2%}")
        print(f"   Sharpe Ratio:        {perf['sharpe_ratio']:>8.2f}")
        print(f"   Sortino Ratio:       {perf['sortino_ratio']:>8.2f}")
        print(f"   Calmar Ratio:        {perf['calmar_ratio']:>8.2f}")
        
        print(f"\nRISK METRICS:")
        print(f"   Maximum Drawdown:    {perf['max_drawdown']:>8.2%}")
        print(f"   VaR (95%):           {risk['var_95']:>8.2%}")
        print(f"   CVaR (95%):          {risk['cvar_95']:>8.2%}")
        print(f"   Max Drawdown Duration: {risk['max_drawdown_duration']:>6.0f} days")
        
        print(f"\nTRADING METRICS:")
        print(f"   Total Trades:        {trading.get('total_trades', 0):>8.0f}")
        print(f"   Win Rate:            {trading.get('win_rate', 0):>8.2%}")
        print(f"   Profit Factor:       {trading.get('profit_factor', 0):>8.2f}")
        print(f"   Avg Trade Return:    {trading.get('avg_trade_return', 0):>8.2%}")
        print(f"   Largest Win:         {trading.get('largest_win', 0):>8.2%}")
        print(f"   Largest Loss:        {trading.get('largest_loss', 0):>8.2%}")
        
        if perf['benchmark_return'] != 0:
            print(f"\nBENCHMARK COMPARISON:")
            print(f"   Strategy Return:     {perf['annualized_return']:>8.2%}")
            print(f"   Benchmark Return:    {perf['benchmark_return']:>8.2%}")
            print(f"   Alpha:               {perf['alpha']:>8.2%}")
            print(f"   Beta:                {perf['beta']:>8.2f}")
            print(f"   Information Ratio:   {perf['information_ratio']:>8.2f}")
        
        print("\n" + "="*80)


def create_sample_backtest_results() -> BacktestResults:
    """Create sample backtest results for testing"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Simulate portfolio performance
    np.random.seed(42)
    daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # Slightly positive mean
    portfolio_values = pd.Series((1 + pd.Series(daily_returns)).cumprod() * 100000, index=dates)
    returns = pd.Series(daily_returns, index=dates)
    
    # Simulate benchmark
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
    
    # Calculate drawdowns
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    
    # Rolling metrics
    rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
    rolling_volatility = returns.rolling(60).std()
    
    # Sample positions
    positions = pd.DataFrame({
        'total_exposure': np.random.uniform(50000, 95000, len(dates))
    }, index=dates)
    
    # Sample trades
    trade_dates = np.random.choice(dates, 50)
    trades = pd.DataFrame({
        'pnl': np.random.normal(0.01, 0.03, 50),
        'entry_price': np.random.uniform(50, 200, 50),
        'exit_price': np.random.uniform(45, 210, 50)
    }, index=trade_dates)
    
    return BacktestResults(
        portfolio_values=portfolio_values,
        returns=returns,
        positions=positions,
        trades=trades,
        benchmark_returns=benchmark_returns,
        drawdowns=drawdowns,
        rolling_sharpe=rolling_sharpe,
        rolling_volatility=rolling_volatility,
        strategy_params={'min_expected_return': 0.02, 'max_positions': 10},
        start_date=dates[0].to_pydatetime(),
        end_date=dates[-1].to_pydatetime(),
        initial_capital=100000
    )


if __name__ == "__main__":
    # Test the report generation
    print("Testing BacktestReport generation...")
    
    # Create sample results
    sample_results = create_sample_backtest_results()
    
    # Generate report
    report = BacktestReport(sample_results)
    metrics = report.generate_report(save_path=".")
    
    print("\nReport generation completed successfully!")