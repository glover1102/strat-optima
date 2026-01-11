"""
Simple performance metrics calculator for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame, 
                     initial_capital: float = 100000) -> Dict[str, float]:
    """
    Calculate key performance metrics from backtest results.
    
    Args:
        equity_curve: Series of equity values over time
        trades_df: DataFrame with trade information
        initial_capital: Starting capital
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {}
    
    # Total Return
    final_equity = equity_curve.iloc[-1]
    metrics['Total Return'] = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Max Drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    metrics['Max Drawdown'] = drawdown.min()
    
    # Win Rate
    if len(trades_df) > 0 and 'pnl' in trades_df.columns:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        metrics['Win Rate'] = (len(winning_trades) / len(trades_df)) * 100
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss != 0 else 0
        
        # Average Win/Loss
        if len(winning_trades) > 0:
            metrics['Avg Win'] = winning_trades['pnl'].mean()
        else:
            metrics['Avg Win'] = 0
            
        losing_trades = trades_df[trades_df['pnl'] < 0]
        if len(losing_trades) > 0:
            metrics['Avg Loss'] = losing_trades['pnl'].mean()
        else:
            metrics['Avg Loss'] = 0
        
        # Number of Trades
        metrics['Total Trades'] = len(trades_df)
        metrics['Winning Trades'] = len(winning_trades)
        metrics['Losing Trades'] = len(losing_trades)
        
    else:
        metrics['Win Rate'] = 0
        metrics['Profit Factor'] = 0
        metrics['Total Trades'] = 0
        metrics['Winning Trades'] = 0
        metrics['Losing Trades'] = 0
        metrics['Avg Win'] = 0
        metrics['Avg Loss'] = 0
    
    # Sharpe Ratio (simplified, annualized)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0 and returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Assuming daily data
        metrics['Sharpe Ratio'] = sharpe
    else:
        metrics['Sharpe Ratio'] = 0
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Print performance metrics in a formatted way.
    
    Args:
        metrics: Dictionary with performance metrics
    """
    print("\n" + "="*60)
    print("BACKTEST PERFORMANCE METRICS")
    print("="*60)
    
    print(f"\nReturn Metrics:")
    print(f"  Total Return:        {metrics['Total Return']:>10.2f}%")
    print(f"  Sharpe Ratio:        {metrics['Sharpe Ratio']:>10.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:        {metrics['Max Drawdown']:>10.2f}%")
    
    print(f"\nTrade Metrics:")
    print(f"  Total Trades:        {metrics['Total Trades']:>10.0f}")
    print(f"  Winning Trades:      {metrics['Winning Trades']:>10.0f}")
    print(f"  Losing Trades:       {metrics['Losing Trades']:>10.0f}")
    print(f"  Win Rate:            {metrics['Win Rate']:>10.2f}%")
    print(f"  Profit Factor:       {metrics['Profit Factor']:>10.2f}")
    print(f"  Avg Win:             ${metrics['Avg Win']:>10.2f}")
    print(f"  Avg Loss:            ${metrics['Avg Loss']:>10.2f}")
    
    print("="*60 + "\n")
