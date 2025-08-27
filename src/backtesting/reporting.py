"""
Reporting module with comprehensive visualizations and export functionality
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import json
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive backtest reports with visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize report generator
        
        Args:
            style: Matplotlib style to use for plots
        """
        self.style = style
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style '{style}' not available, using default")
        
        # Set color palette
        self.colors = {
            'portfolio': '#2E86C1',
            'benchmark': '#E74C3C',
            'drawdown': '#E67E22',
            'positive': '#27AE60',
            'negative': '#E74C3C',
            'neutral': '#7F8C8D'
        }
        
        sns.set_palette("husl")
        
        logger.info("ReportGenerator initialized")
    
    def generate_full_report(self, backtest_results: Dict[str, Any], 
                           output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate complete backtest report with all visualizations
        
        Args:
            backtest_results: Results from BacktestEngine.run()
            output_dir: Directory to save reports
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Generating comprehensive backtest report")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"backtest_report_{timestamp}"
        
        generated_files = {}
        
        try:
            # Generate PDF report
            pdf_path = output_path / f"{base_name}.pdf"
            self._generate_pdf_report(backtest_results, pdf_path)
            generated_files['pdf'] = str(pdf_path)
            
            # Generate HTML dashboard
            html_path = output_path / f"{base_name}.html"
            self._generate_html_dashboard(backtest_results, html_path)
            generated_files['html'] = str(html_path)
            
            # Export data files
            csv_path = output_path / f"{base_name}_data.csv"
            json_path = output_path / f"{base_name}_metrics.json"
            
            self._export_csv_data(backtest_results, csv_path)
            self._export_json_metrics(backtest_results, json_path)
            
            generated_files['csv'] = str(csv_path)
            generated_files['json'] = str(json_path)
            
            logger.info(f"Report generation completed. Files saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
        
        return generated_files
    
    def _generate_pdf_report(self, results: Dict[str, Any], output_path: Path):
        """Generate comprehensive PDF report"""
        with PdfPages(output_path) as pdf:
            # Page 1: Executive Summary
            self._create_summary_page(results, pdf)
            
            # Page 2: Equity Curve and Performance
            self._create_performance_page(results, pdf)
            
            # Page 3: Risk Analysis
            self._create_risk_page(results, pdf)
            
            # Page 4: Trade Analysis
            self._create_trade_analysis_page(results, pdf)
            
            # Page 5: Monthly Returns Heatmap
            self._create_monthly_returns_page(results, pdf)
            
            # Page 6: Rolling Metrics
            self._create_rolling_metrics_page(results, pdf)
    
    def _create_summary_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create executive summary page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Backtest Executive Summary', fontsize=16, fontweight='bold')
        
        metrics = results.get('metrics', {})
        
        # Key metrics table
        key_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0):.2%}"),
            ('Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Maximum Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Volatility', f"{metrics.get('volatility', 0):.2%}"),
            ('Total Trades', f"{metrics.get('total_trades', 0):,}")
        ]
        
        # Create table
        ax1.axis('tight')
        ax1.axis('off')
        table_data = [[metric, value] for metric, value in key_metrics]
        table = ax1.table(cellText=table_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax1.set_title('Key Performance Metrics')
        
        # Equity curve thumbnail
        if 'equity_curve' in results:
            dates = list(results['equity_curve'].keys())
            values = list(results['equity_curve'].values())
            dates_parsed = [pd.to_datetime(d) for d in dates]
            
            ax2.plot(dates_parsed, values, color=self.colors['portfolio'], linewidth=2)
            ax2.set_title('Equity Curve')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Risk metrics visualization
        risk_metrics = ['volatility', 'max_drawdown', 'sharpe_ratio']
        risk_values = [abs(metrics.get(metric, 0)) for metric in risk_metrics]
        risk_labels = ['Volatility', 'Max Drawdown', 'Sharpe Ratio']
        
        colors = [self.colors['negative'] if x > 0.15 else self.colors['positive'] for x in risk_values[:2]]
        colors.append(self.colors['positive'] if risk_values[2] > 1 else self.colors['negative'])
        
        ax3.barh(risk_labels, risk_values, color=colors)
        ax3.set_title('Risk Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Configuration summary
        config_text = f"""
Configuration Summary:
• Initial Capital: ${results.get('config', {}).get('initial_capital', 0):,.0f}
• Period: {results.get('config', {}).get('start_date', 'N/A')} to {results.get('config', {}).get('end_date', 'N/A')}
• Final Value: ${metrics.get('final_value', 0):,.0f}
• Strategy: ML-based prediction strategy
        """
        
        ax4.text(0.05, 0.95, config_text.strip(), transform=ax4.transAxes,
                verticalalignment='top', fontsize=9, family='monospace')
        ax4.axis('off')
        ax4.set_title('Configuration')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_performance_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create performance analysis page"""
        fig = plt.figure(figsize=(11, 8.5))
        
        # Main equity curve
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        
        if 'equity_curve' in results:
            dates = list(results['equity_curve'].keys())
            values = list(results['equity_curve'].values())
            dates_parsed = [pd.to_datetime(d) for d in dates]
            
            ax1.plot(dates_parsed, values, color=self.colors['portfolio'], 
                    linewidth=2, label='Portfolio')
            
            # Add initial capital line
            initial_capital = results.get('config', {}).get('initial_capital', values[0])
            ax1.axhline(y=initial_capital, color=self.colors['benchmark'], 
                       linestyle='--', alpha=0.7, label='Initial Capital')
            
            ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Underwater plot (drawdowns)
        ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        self._plot_underwater(results, ax2)
        
        # Returns distribution
        ax3 = plt.subplot2grid((3, 2), (2, 0))
        if 'daily_returns' in results:
            daily_returns = list(results['daily_returns'].values())
            ax3.hist(daily_returns, bins=50, alpha=0.7, color=self.colors['portfolio'], 
                    edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Daily Return')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Performance metrics comparison
        ax4 = plt.subplot2grid((3, 2), (2, 1))
        metrics = results.get('metrics', {})
        perf_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        perf_values = [metrics.get('total_return', 0), 
                      metrics.get('sharpe_ratio', 0), 
                      -abs(metrics.get('max_drawdown', 0))]  # Negative for drawdown
        perf_labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
        
        colors = [self.colors['positive'], 
                 self.colors['positive'] if perf_values[1] > 0 else self.colors['negative'],
                 self.colors['negative']]
        
        bars = ax4.bar(perf_labels, perf_values, color=colors)
        ax4.set_title('Key Performance Metrics')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, perf_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_risk_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create risk analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        # VaR visualization
        if 'daily_returns' in results:
            daily_returns = list(results['daily_returns'].values())
            sorted_returns = sorted(daily_returns)
            
            # Calculate VaR levels
            var_95 = np.percentile(sorted_returns, 5)
            var_99 = np.percentile(sorted_returns, 1)
            
            ax1.hist(daily_returns, bins=50, alpha=0.7, color=self.colors['portfolio'])
            ax1.axvline(x=var_95, color='orange', linestyle='--', 
                       label=f'VaR 95%: {var_95:.2%}')
            ax1.axvline(x=var_99, color='red', linestyle='--', 
                       label=f'VaR 99%: {var_99:.2%}')
            ax1.set_title('Value at Risk Analysis')
            ax1.set_xlabel('Daily Return')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        if 'equity_curve' in results:
            dates = list(results['equity_curve'].keys())
            values = list(results['equity_curve'].values())
            dates_parsed = [pd.to_datetime(d) for d in dates]
            
            # Calculate rolling volatility
            returns_series = pd.Series(values, index=dates_parsed).pct_change().dropna()
            rolling_vol = returns_series.rolling(window=30).std() * np.sqrt(252)
            
            ax2.plot(rolling_vol.index, rolling_vol.values, 
                    color=self.colors['drawdown'], linewidth=2)
            ax2.set_title('Rolling 30-Day Volatility')
            ax2.set_ylabel('Annualized Volatility')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Risk-Return Scatter (placeholder)
        metrics = results.get('metrics', {})
        ax3.scatter([metrics.get('volatility', 0)], [metrics.get('annualized_return', 0)], 
                   s=100, color=self.colors['portfolio'], label='Portfolio')
        ax3.set_xlabel('Volatility (Annualized)')
        ax3.set_ylabel('Return (Annualized)')
        ax3.set_title('Risk-Return Profile')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Risk metrics table
        risk_data = [
            ['Volatility', f"{metrics.get('volatility', 0):.2%}"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
            ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"],
            ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"],
            ['VaR 95%', f"{metrics.get('var_95', 0):.2%}"]
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=risk_data, 
                         colLabels=['Risk Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax4.set_title('Risk Metrics Summary')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_trade_analysis_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create trade analysis page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        trades_log = results.get('trades_log', [])
        
        if trades_log:
            # Trade volume over time
            trade_df = pd.DataFrame(trades_log)
            trade_df['date'] = pd.to_datetime(trade_df['date'])
            
            # Daily trade counts
            daily_trades = trade_df.groupby(trade_df['date'].dt.date).size()
            
            ax1.bar(range(len(daily_trades)), daily_trades.values, 
                   color=self.colors['portfolio'], alpha=0.7)
            ax1.set_title('Daily Trade Count')
            ax1.set_xlabel('Trading Day')
            ax1.set_ylabel('Number of Trades')
            ax1.grid(True, alpha=0.3)
            
            # Trade size distribution
            ax2.hist(trade_df['shares'], bins=30, alpha=0.7, 
                    color=self.colors['portfolio'], edgecolor='black')
            ax2.set_title('Trade Size Distribution')
            ax2.set_xlabel('Shares per Trade')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Buy vs Sell trades
            trade_types = trade_df['type'].value_counts()
            colors = [self.colors['positive'], self.colors['negative']]
            ax3.pie(trade_types.values, labels=trade_types.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax3.set_title('Trade Type Distribution')
            
            # Trading costs analysis
            if 'commission' in trade_df.columns:
                total_commission = trade_df['commission'].sum()
                avg_commission = trade_df['commission'].mean()
                
                cost_text = f"""
Trading Costs Summary:
• Total Trades: {len(trade_df):,}
• Total Commission: ${total_commission:,.2f}
• Average Commission: ${avg_commission:.2f}
• Commission as % of Volume: {(total_commission / trade_df['execution_price'].sum() * 100):.3f}%
                """
                
                ax4.text(0.05, 0.95, cost_text.strip(), transform=ax4.transAxes,
                        verticalalignment='top', fontsize=9, family='monospace')
                ax4.set_title('Trading Costs')
        else:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No trade data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trade Analysis')
        
        ax4.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_monthly_returns_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create monthly returns heatmap page"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Monthly Returns Analysis', fontsize=16, fontweight='bold')
        
        if 'daily_returns' in results:
            # Convert daily returns to monthly
            daily_returns = results['daily_returns']
            dates = [pd.to_datetime(d) for d in daily_returns.keys()]
            returns = list(daily_returns.values())
            
            returns_series = pd.Series(returns, index=dates)
            monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            if len(monthly_returns) > 0:
                # Create monthly returns table
                monthly_df = monthly_returns.to_frame('Returns')
                monthly_df['Year'] = monthly_df.index.year
                monthly_df['Month'] = monthly_df.index.strftime('%b')
                
                # Pivot for heatmap
                heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Returns')
                
                # Reorder columns by month
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                heatmap_data = heatmap_data.reindex(columns=month_order)
                
                # Create heatmap
                sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                           center=0, ax=ax1, cbar_kws={'label': 'Monthly Return'})
                ax1.set_title('Monthly Returns Heatmap')
                
                # Monthly returns distribution
                ax2.hist(monthly_returns.values, bins=20, alpha=0.7, 
                        color=self.colors['portfolio'], edgecolor='black')
                ax2.axvline(x=monthly_returns.mean(), color='red', linestyle='--', 
                           label=f'Mean: {monthly_returns.mean():.2%}')
                ax2.set_title('Monthly Returns Distribution')
                ax2.set_xlabel('Monthly Return')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient data for monthly analysis', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'Insufficient data for monthly analysis', 
                        ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_rolling_metrics_page(self, results: Dict[str, Any], pdf: PdfPages):
        """Create rolling metrics page"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold')
        
        if 'daily_returns' in results:
            daily_returns = results['daily_returns']
            dates = [pd.to_datetime(d) for d in daily_returns.keys()]
            returns = list(daily_returns.values())
            
            returns_series = pd.Series(returns, index=dates)
            
            # Rolling Sharpe ratio (60-day)
            if len(returns_series) >= 60:
                rolling_sharpe = returns_series.rolling(60).mean() / returns_series.rolling(60).std() * np.sqrt(252)
                ax1.plot(rolling_sharpe.index, rolling_sharpe.values, 
                        color=self.colors['portfolio'], linewidth=2)
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.set_title('60-Day Rolling Sharpe Ratio')
                ax1.set_ylabel('Sharpe Ratio')
                ax1.grid(True, alpha=0.3)
            
            # Rolling max drawdown (60-day)
            if len(returns_series) >= 60:
                cumulative = (1 + returns_series).cumprod()
                rolling_max = cumulative.rolling(60).max()
                rolling_dd = ((cumulative - rolling_max) / rolling_max).rolling(60).min()
                
                ax2.fill_between(rolling_dd.index, 0, rolling_dd.values, 
                               color=self.colors['drawdown'], alpha=0.7)
                ax2.set_title('60-Day Rolling Max Drawdown')
                ax2.set_ylabel('Drawdown')
                ax2.grid(True, alpha=0.3)
            
            # Rolling volatility (30-day)
            if len(returns_series) >= 30:
                rolling_vol = returns_series.rolling(30).std() * np.sqrt(252)
                ax3.plot(rolling_vol.index, rolling_vol.values, 
                        color=self.colors['drawdown'], linewidth=2)
                ax3.set_title('30-Day Rolling Volatility')
                ax3.set_ylabel('Annualized Volatility')
                ax3.grid(True, alpha=0.3)
            
            # Rolling returns (30-day)
            if len(returns_series) >= 30:
                rolling_returns = returns_series.rolling(30).apply(lambda x: (1 + x).prod() - 1)
                ax4.plot(rolling_returns.index, rolling_returns.values, 
                        color=self.colors['portfolio'], linewidth=2)
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.set_title('30-Day Rolling Returns')
                ax4.set_ylabel('30-Day Return')
                ax4.grid(True, alpha=0.3)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _plot_underwater(self, results: Dict[str, Any], ax):
        """Create underwater plot (drawdown chart)"""
        if 'equity_curve' in results:
            dates = list(results['equity_curve'].keys())
            values = list(results['equity_curve'].values())
            dates_parsed = [pd.to_datetime(d) for d in dates]
            
            # Calculate drawdowns
            cumulative = np.array(values)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            ax.fill_between(dates_parsed, 0, drawdown, color=self.colors['drawdown'], alpha=0.7)
            ax.set_title('Drawdown (Underwater Plot)')
            ax.set_ylabel('Drawdown')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add max drawdown line
            max_dd = drawdown.min()
            ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7, 
                      label=f'Max Drawdown: {max_dd:.2%}')
            ax.legend()
    
    def _generate_html_dashboard(self, results: Dict[str, Any], output_path: Path):
        """Generate interactive HTML dashboard"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { 
                    display: inline-block; 
                    background: #f0f0f0; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 5px; 
                    min-width: 200px;
                }
                .metric-value { font-size: 24px; font-weight: bold; color: #2E86C1; }
                .chart-container { margin: 20px 0; }
                h1, h2 { color: #2C3E50; }
            </style>
        </head>
        <body>
            <h1>Backtest Results Dashboard</h1>
            
            <div id="metrics-summary">
                <h2>Key Metrics</h2>
                {metrics_cards}
            </div>
            
            <div class="chart-container">
                <div id="equity-curve" style="height: 400px;"></div>
            </div>
            
            <div class="chart-container">
                <div id="drawdown-chart" style="height: 300px;"></div>
            </div>
            
            <div class="chart-container">
                <div id="returns-histogram" style="height: 300px;"></div>
            </div>
            
            <script>
                {javascript_code}
            </script>
        </body>
        </html>
        """
        
        # Generate metrics cards
        metrics = results.get('metrics', {})
        metrics_cards = ""
        key_metrics = [
            ('Total Return', metrics.get('total_return', 0), '{:.2%}'),
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), '{:.2f}'),
            ('Max Drawdown', metrics.get('max_drawdown', 0), '{:.2%}'),
            ('Volatility', metrics.get('volatility', 0), '{:.2%}')
        ]
        
        for name, value, fmt in key_metrics:
            formatted_value = fmt.format(value)
            metrics_cards += f'''
            <div class="metric-card">
                <div>{name}</div>
                <div class="metric-value">{formatted_value}</div>
            </div>
            '''
        
        # Generate JavaScript for interactive charts
        javascript_code = self._generate_plotly_charts(results)
        
        # Write HTML file
        html_content = html_template.format(
            metrics_cards=metrics_cards,
            javascript_code=javascript_code
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_plotly_charts(self, results: Dict[str, Any]) -> str:
        """Generate Plotly JavaScript code for interactive charts"""
        
        # Equity curve data
        equity_data = "[]"
        drawdown_data = "[]"
        returns_data = "[]"
        
        if 'equity_curve' in results:
            dates = list(results['equity_curve'].keys())
            values = list(results['equity_curve'].values())
            
            equity_data = f"{{x: {json.dumps(dates)}, y: {json.dumps(values)}, type: 'scatter', name: 'Portfolio Value'}}"
            
            # Calculate drawdowns for underwater plot
            cumulative = np.array(values)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = ((cumulative - running_max) / running_max).tolist()
            
            drawdown_data = f"{{x: {json.dumps(dates)}, y: {json.dumps(drawdown)}, fill: 'tonexty', type: 'scatter', name: 'Drawdown'}}"
        
        if 'daily_returns' in results:
            returns = list(results['daily_returns'].values())
            returns_data = json.dumps(returns)
        
        javascript = f"""
        // Equity Curve
        Plotly.newPlot('equity-curve', [{equity_data}], {{
            title: 'Portfolio Equity Curve',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Portfolio Value ($)'}}
        }});
        
        // Drawdown Chart
        Plotly.newPlot('drawdown-chart', [{drawdown_data}], {{
            title: 'Drawdown Chart',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Drawdown', tickformat: '.2%'}}
        }});
        
        // Returns Histogram
        Plotly.newPlot('returns-histogram', [{{
            x: {returns_data},
            type: 'histogram',
            nbinsx: 50,
            name: 'Daily Returns'
        }}], {{
            title: 'Daily Returns Distribution',
            xaxis: {{title: 'Daily Return', tickformat: '.2%'}},
            yaxis: {{title: 'Frequency'}}
        }});
        """
        
        return javascript
    
    def _export_csv_data(self, results: Dict[str, Any], output_path: Path):
        """Export portfolio history to CSV"""
        portfolio_history = results.get('portfolio_history', [])
        
        if portfolio_history:
            # Convert to DataFrame
            df_data = []
            for record in portfolio_history:
                row = {
                    'date': record['date'],
                    'portfolio_value': record['portfolio_value'],
                    'cash': record['cash'],
                    'positions_value': record.get('positions_value', 0),
                    'num_positions': record.get('num_positions', 0),
                    'daily_return': record.get('daily_return', 0)
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
    
    def _export_json_metrics(self, results: Dict[str, Any], output_path: Path):
        """Export metrics to JSON"""
        # Prepare JSON-serializable data
        export_data = {
            'config': results.get('config', {}),
            'metrics': results.get('metrics', {}),
            'generation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_trades': results.get('metrics', {}).get('total_trades', 0),
                'final_value': results.get('metrics', {}).get('final_value', 0),
                'total_return': results.get('metrics', {}).get('total_return', 0)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def create_quick_summary(self, results: Dict[str, Any]) -> str:
        """Create a quick text summary of results"""
        metrics = results.get('metrics', {})
        
        summary = f"""
BACKTEST SUMMARY
================
Period: {results.get('config', {}).get('start_date', 'N/A')} to {results.get('config', {}).get('end_date', 'N/A')}
Initial Capital: ${results.get('config', {}).get('initial_capital', 0):,.2f}
Final Value: ${metrics.get('final_value', 0):,.2f}

PERFORMANCE METRICS
===================
Total Return: {metrics.get('total_return', 0):.2%}
Annualized Return: {metrics.get('annualized_return', 0):.2%}
Volatility: {metrics.get('volatility', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}

TRADING ACTIVITY
================
Total Trades: {metrics.get('total_trades', 0):,}
Buy Trades: {metrics.get('buy_trades', 0):,}
Sell Trades: {metrics.get('sell_trades', 0):,}
        """
        
        return summary.strip()