"""
Reporting Utilities for Strategy Analysis

Generates comprehensive performance reports and analysis summaries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from pathlib import Path

from ..optimization.metrics import PerformanceMetrics, metrics_to_dict

logger = logging.getLogger(__name__)


def generate_performance_report(metrics: PerformanceMetrics,
                               trades_df: Optional[pd.DataFrame] = None,
                               title: str = "Strategy Performance Report") -> str:
    """
    Generate comprehensive text performance report.
    
    Args:
        metrics: PerformanceMetrics object
        trades_df: Optional DataFrame with trade details
        title: Report title
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append(f"{title:^80}")
    report.append("=" * 80)
    report.append("")
    
    # Returns section
    report.append("RETURNS")
    report.append("-" * 80)
    report.append(f"Total Return:        {metrics.total_return:>12.2%}")
    report.append(f"Annual Return:       {metrics.annual_return:>12.2%}")
    report.append(f"Cumulative Return:   {metrics.cumulative_return:>12.2%}")
    report.append("")
    
    # Risk section
    report.append("RISK METRICS")
    report.append("-" * 80)
    report.append(f"Volatility:          {metrics.volatility:>12.2%}")
    report.append(f"Max Drawdown:        {metrics.max_drawdown:>12.2%}")
    report.append(f"Avg Drawdown:        {metrics.avg_drawdown:>12.2%}")
    report.append(f"Max DD Duration:     {metrics.max_drawdown_duration:>12} periods")
    report.append("")
    
    # Risk-adjusted returns
    report.append("RISK-ADJUSTED RETURNS")
    report.append("-" * 80)
    report.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:>12.2f}")
    report.append(f"Sortino Ratio:       {metrics.sortino_ratio:>12.2f}")
    report.append(f"Calmar Ratio:        {metrics.calmar_ratio:>12.2f}")
    report.append(f"Recovery Factor:     {metrics.recovery_factor:>12.2f}")
    report.append("")
    
    # Trade statistics
    report.append("TRADE STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Trades:        {metrics.total_trades:>12}")
    report.append(f"Winning Trades:      {metrics.winning_trades:>12}")
    report.append(f"Losing Trades:       {metrics.losing_trades:>12}")
    report.append(f"Win Rate:            {metrics.win_rate:>12.2%}")
    report.append("")
    
    # Profit metrics
    report.append("PROFIT METRICS")
    report.append("-" * 80)
    report.append(f"Profit Factor:       {metrics.profit_factor:>12.2f}")
    report.append(f"Expectancy:          {metrics.expectancy:>12.2f}")
    report.append(f"Avg Win:             {metrics.avg_win:>12.2f}")
    report.append(f"Avg Loss:            {metrics.avg_loss:>12.2f}")
    report.append(f"Avg Trade:           {metrics.avg_trade:>12.2f}")
    report.append(f"Largest Win:         {metrics.largest_win:>12.2f}")
    report.append(f"Largest Loss:        {metrics.largest_loss:>12.2f}")
    report.append("")
    
    # Exposure
    report.append("EXPOSURE")
    report.append("-" * 80)
    report.append(f"Time in Market:      {metrics.exposure_time:>12.2%}")
    report.append("")
    
    # Trade details if provided
    if trades_df is not None and not trades_df.empty:
        report.append("TRADE DETAILS")
        report.append("-" * 80)
        
        # Exit reasons
        if 'exit_reason' in trades_df.columns:
            exit_reasons = trades_df['exit_reason'].value_counts()
            report.append("Exit Reasons:")
            for reason, count in exit_reasons.items():
                pct = count / len(trades_df) * 100
                report.append(f"  {reason:<20} {count:>6} ({pct:>5.1f}%)")
        
        report.append("")
        
        # Direction breakdown
        if 'direction' in trades_df.columns:
            long_trades = trades_df[trades_df['direction'] == 1]
            short_trades = trades_df[trades_df['direction'] == -1]
            
            report.append("Direction Breakdown:")
            report.append(f"  Long Trades:       {len(long_trades):>6} " +
                         f"(Win Rate: {(long_trades['pnl'] > 0).mean():.2%})")
            report.append(f"  Short Trades:      {len(short_trades):>6} " +
                         f"(Win Rate: {(short_trades['pnl'] > 0).mean():.2%})")
        
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_walk_forward_report(periods: List, wfe: float,
                                 param_stability: Optional[pd.DataFrame] = None) -> str:
    """
    Generate walk-forward optimization report.
    
    Args:
        periods: List of WalkForwardPeriod objects
        wfe: Walk-forward efficiency
        param_stability: Optional parameter stability DataFrame
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append(f"{'WALK-FORWARD OPTIMIZATION REPORT':^80}")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Number of Periods:   {len(periods):>12}")
    report.append(f"Walk-Forward Efficiency: {wfe:>8.2f}")
    report.append("")
    
    # Aggregate performance
    is_returns = [p.train_metrics.total_return for p in periods if p.train_metrics]
    oos_returns = [p.test_metrics.total_return for p in periods if p.test_metrics]
    is_sharpe = [p.train_metrics.sharpe_ratio for p in periods if p.train_metrics]
    oos_sharpe = [p.test_metrics.sharpe_ratio for p in periods if p.test_metrics]
    
    report.append("AGGREGATE PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Avg IS Return:       {np.mean(is_returns):>12.2%}")
    report.append(f"Avg OOS Return:      {np.mean(oos_returns):>12.2%}")
    report.append(f"Avg IS Sharpe:       {np.mean(is_sharpe):>12.2f}")
    report.append(f"Avg OOS Sharpe:      {np.mean(oos_sharpe):>12.2f}")
    report.append("")
    
    # Period-by-period results
    report.append("PERIOD-BY-PERIOD RESULTS")
    report.append("-" * 80)
    report.append(f"{'Period':<12} {'IS Return':>12} {'OOS Return':>12} " +
                 f"{'IS Sharpe':>12} {'OOS Sharpe':>12}")
    report.append("-" * 80)
    
    for i, period in enumerate(periods):
        period_label = period.test_start.strftime('%Y-%m')
        is_ret = period.train_metrics.total_return if period.train_metrics else 0
        oos_ret = period.test_metrics.total_return if period.test_metrics else 0
        is_sh = period.train_metrics.sharpe_ratio if period.train_metrics else 0
        oos_sh = period.test_metrics.sharpe_ratio if period.test_metrics else 0
        
        report.append(f"{period_label:<12} {is_ret:>11.2%} {oos_ret:>11.2%} " +
                     f"{is_sh:>11.2f} {oos_sh:>11.2f}")
    
    report.append("")
    
    # Parameter stability
    if param_stability is not None:
        report.append("PARAMETER STABILITY")
        report.append("-" * 80)
        report.append(f"{'Parameter':<20} {'Mean':>10} {'Std':>10} " +
                     f"{'Min':>10} {'Max':>10} {'CV':>10}")
        report.append("-" * 80)
        
        for param in param_stability.index:
            row = param_stability.loc[param]
            report.append(f"{param:<20} {row['mean']:>10.3f} {row['std']:>10.3f} " +
                         f"{row['min']:>10.3f} {row['max']:>10.3f} {row['cv']:>10.3f}")
        
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def create_metrics_dataframe(metrics_list: List[PerformanceMetrics],
                             labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create DataFrame from list of metrics.
    
    Args:
        metrics_list: List of PerformanceMetrics objects
        labels: Optional labels for each metrics set
        
    Returns:
        DataFrame with metrics
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_list))]
    
    data = []
    for metrics in metrics_list:
        data.append(metrics_to_dict(metrics))
    
    df = pd.DataFrame(data, index=labels)
    return df


def export_trades_to_csv(trades_df: pd.DataFrame, filepath: str) -> None:
    """
    Export trades to CSV file.
    
    Args:
        trades_df: DataFrame with trade information
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    trades_df.to_csv(filepath, index=False)
    logger.info(f"Trades exported to {filepath}")


def export_metrics_to_csv(metrics: PerformanceMetrics, filepath: str) -> None:
    """
    Export metrics to CSV file.
    
    Args:
        metrics: PerformanceMetrics object
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_dict = metrics_to_dict(metrics)
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filepath, index=False)
    
    logger.info(f"Metrics exported to {filepath}")


def create_summary_dashboard(periods: List, trades_df: pd.DataFrame,
                            equity: pd.Series) -> Dict:
    """
    Create summary dashboard data.
    
    Args:
        periods: List of WalkForwardPeriod objects
        trades_df: DataFrame with all trades
        equity: Equity curve
        
    Returns:
        Dictionary with dashboard data
    """
    # Calculate overall metrics
    from ..optimization.metrics import calculate_all_metrics
    overall_metrics = calculate_all_metrics(trades_df)
    
    # WFE
    wfe = 0.0
    if periods:
        is_returns = [p.train_metrics.total_return for p in periods if p.train_metrics]
        oos_returns = [p.test_metrics.total_return for p in periods if p.test_metrics]
        if sum(is_returns) > 0:
            wfe = sum(oos_returns) / sum(is_returns)
    
    # Key statistics
    dashboard = {
        'overview': {
            'total_return': overall_metrics.total_return,
            'annual_return': overall_metrics.annual_return,
            'sharpe_ratio': overall_metrics.sharpe_ratio,
            'max_drawdown': overall_metrics.max_drawdown,
            'win_rate': overall_metrics.win_rate,
            'profit_factor': overall_metrics.profit_factor,
        },
        'walk_forward': {
            'num_periods': len(periods),
            'wfe': wfe,
            'avg_oos_return': np.mean([p.test_metrics.total_return for p in periods 
                                       if p.test_metrics]) if periods else 0,
            'avg_oos_sharpe': np.mean([p.test_metrics.sharpe_ratio for p in periods 
                                       if p.test_metrics]) if periods else 0,
        },
        'trades': {
            'total_trades': len(trades_df),
            'winning_trades': (trades_df['pnl'] > 0).sum() if not trades_df.empty else 0,
            'avg_trade_pnl': trades_df['pnl'].mean() if not trades_df.empty else 0,
            'best_trade': trades_df['pnl'].max() if not trades_df.empty else 0,
            'worst_trade': trades_df['pnl'].min() if not trades_df.empty else 0,
        }
    }
    
    return dashboard


def save_report_to_file(report: str, filepath: str) -> None:
    """
    Save text report to file.
    
    Args:
        report: Report string
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {filepath}")


def generate_markdown_report(metrics: PerformanceMetrics,
                            trades_df: Optional[pd.DataFrame] = None,
                            periods: Optional[List] = None,
                            title: str = "Strategy Analysis Report") -> str:
    """
    Generate markdown-formatted report.
    
    Args:
        metrics: PerformanceMetrics object
        trades_df: Optional DataFrame with trades
        periods: Optional list of WalkForwardPeriod objects
        title: Report title
        
    Returns:
        Markdown-formatted report string
    """
    report = []
    report.append(f"# {title}")
    report.append("")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report.append("")
    
    # Key metrics table
    report.append("## Key Metrics")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total Return | {metrics.total_return:.2%} |")
    report.append(f"| Annual Return | {metrics.annual_return:.2%} |")
    report.append(f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |")
    report.append(f"| Max Drawdown | {metrics.max_drawdown:.2%} |")
    report.append(f"| Win Rate | {metrics.win_rate:.2%} |")
    report.append(f"| Profit Factor | {metrics.profit_factor:.2f} |")
    report.append("")
    
    # Trade statistics
    report.append("## Trade Statistics")
    report.append("")
    report.append("| Statistic | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Total Trades | {metrics.total_trades} |")
    report.append(f"| Winning Trades | {metrics.winning_trades} |")
    report.append(f"| Losing Trades | {metrics.losing_trades} |")
    report.append(f"| Average Win | {metrics.avg_win:.2f} |")
    report.append(f"| Average Loss | {metrics.avg_loss:.2f} |")
    report.append("")
    
    # Walk-forward results
    if periods:
        report.append("## Walk-Forward Results")
        report.append("")
        report.append("| Period | IS Return | OOS Return | IS Sharpe | OOS Sharpe |")
        report.append("|--------|-----------|------------|-----------|------------|")
        
        for period in periods:
            period_label = period.test_start.strftime('%Y-%m')
            is_ret = period.train_metrics.total_return if period.train_metrics else 0
            oos_ret = period.test_metrics.total_return if period.test_metrics else 0
            is_sh = period.train_metrics.sharpe_ratio if period.train_metrics else 0
            oos_sh = period.test_metrics.sharpe_ratio if period.test_metrics else 0
            
            report.append(f"| {period_label} | {is_ret:.2%} | {oos_ret:.2%} | " +
                         f"{is_sh:.2f} | {oos_sh:.2f} |")
        
        report.append("")
    
    return "\n".join(report)
