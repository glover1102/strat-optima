"""
Performance Metrics Calculations

Comprehensive performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics."""
    
    # Returns
    total_return: float
    annual_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    
    # Exposure
    exposure_time: float  # Percentage of time in market
    
    # Other
    recovery_factor: float
    expectancy: float


def calculate_returns(trades_df: pd.DataFrame, 
                     initial_capital: float = 10000.0) -> pd.Series:
    """
    Calculate equity curve from trades.
    
    Args:
        trades_df: DataFrame with trade results (must have 'pnl' column)
        initial_capital: Starting capital
        
    Returns:
        Series of equity values over time
    """
    if trades_df.empty:
        return pd.Series([initial_capital])
    
    # Create equity curve
    equity = initial_capital + trades_df['pnl'].cumsum()
    equity = pd.concat([pd.Series([initial_capital]), equity])
    equity.index = range(len(equity))
    
    return equity


def calculate_drawdown_series(equity: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.
    
    Args:
        equity: Equity curve series
        
    Returns:
        Drawdown series (negative values)
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / returns.std())
    
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (using downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
    
    return sortino


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        annual_return: Annualized return
        max_drawdown: Maximum drawdown (positive value)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    return annual_return / max_drawdown


def calculate_max_drawdown(equity: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity: Equity curve series
        
    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()
    
    # Find start of drawdown
    start_idx = (equity[:end_idx] == running_max[:end_idx]).idxmax()
    
    return abs(max_dd), start_idx, end_idx


def calculate_avg_drawdown(equity: pd.Series) -> float:
    """
    Calculate average drawdown.
    
    Args:
        equity: Equity curve series
        
    Returns:
        Average drawdown
    """
    drawdown = calculate_drawdown_series(equity)
    negative_dd = drawdown[drawdown < 0]
    
    if len(negative_dd) == 0:
        return 0.0
    
    return abs(negative_dd.mean())


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades_df: DataFrame with 'pnl' column
        
    Returns:
        Profit factor
    """
    if trades_df.empty:
        return 0.0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_win_rate(trades_df: pd.DataFrame) -> float:
    """
    Calculate win rate.
    
    Args:
        trades_df: DataFrame with 'pnl' column
        
    Returns:
        Win rate (0 to 1)
    """
    if trades_df.empty:
        return 0.0
    
    winning_trades = (trades_df['pnl'] > 0).sum()
    total_trades = len(trades_df)
    
    return winning_trades / total_trades if total_trades > 0 else 0.0


def calculate_expectancy(trades_df: pd.DataFrame) -> float:
    """
    Calculate expectancy (average expected profit per trade).
    
    Args:
        trades_df: DataFrame with 'pnl' column
        
    Returns:
        Expectancy
    """
    if trades_df.empty:
        return 0.0
    
    return trades_df['pnl'].mean()


def calculate_recovery_factor(total_return: float, max_drawdown: float) -> float:
    """
    Calculate recovery factor (total return / max drawdown).
    
    Args:
        total_return: Total return
        max_drawdown: Maximum drawdown
        
    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return 0.0
    
    return total_return / max_drawdown


def calculate_all_metrics(trades_df: pd.DataFrame,
                         equity_curve: Optional[pd.Series] = None,
                         initial_capital: float = 10000.0,
                         risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> PerformanceMetrics:
    """
    Calculate all performance metrics from trades.
    
    Args:
        trades_df: DataFrame with trade information
        equity_curve: Optional pre-calculated equity curve
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        PerformanceMetrics object with all metrics
    """
    # Handle empty trades
    if trades_df.empty:
        return PerformanceMetrics(
            total_return=0.0,
            annual_return=0.0,
            cumulative_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            exposure_time=0.0,
            recovery_factor=0.0,
            expectancy=0.0
        )
    
    # Calculate equity curve
    if equity_curve is None:
        equity_curve = calculate_returns(trades_df, initial_capital)
    
    # Returns
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    
    # Calculate period returns for Sharpe/Sortino
    returns = equity_curve.pct_change().dropna()
    
    # Annualize return
    if len(trades_df) > 0 and 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
        total_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        years = total_days / 365.25 if total_days > 0 else 1
    else:
        years = len(trades_df) / periods_per_year if len(trades_df) > 0 else 1
    
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0.0
    
    # Drawdown metrics
    max_dd, dd_start, dd_end = calculate_max_drawdown(equity_curve)
    avg_dd = calculate_avg_drawdown(equity_curve)
    max_dd_duration = dd_end - dd_start if dd_end > dd_start else 0
    
    # Risk-adjusted returns
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = calculate_calmar_ratio(annual_return, max_dd)
    
    # Trade statistics
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl'] > 0).sum()
    losing_trades = (trades_df['pnl'] < 0).sum()
    win_rate = calculate_win_rate(trades_df)
    
    # Profit metrics
    profit_factor = calculate_profit_factor(trades_df)
    
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    avg_trade = trades_df['pnl'].mean()
    largest_win = wins.max() if len(wins) > 0 else 0.0
    largest_loss = losses.min() if len(losses) > 0 else 0.0
    
    # Exposure time
    if 'duration' in trades_df.columns:
        total_duration = trades_df['duration'].sum()
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            total_time = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).total_seconds() / 3600
            exposure_time = total_duration / total_time if total_time > 0 else 0.0
        else:
            exposure_time = 0.0
    else:
        exposure_time = 0.0
    
    # Recovery factor
    recovery_factor = calculate_recovery_factor(total_return, max_dd)
    
    # Expectancy
    expectancy = calculate_expectancy(trades_df)
    
    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        cumulative_return=total_return,
        volatility=volatility,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_trade=avg_trade,
        largest_win=largest_win,
        largest_loss=largest_loss,
        exposure_time=exposure_time,
        recovery_factor=recovery_factor,
        expectancy=expectancy
    )


def metrics_to_dict(metrics: PerformanceMetrics) -> Dict[str, float]:
    """
    Convert PerformanceMetrics to dictionary.
    
    Args:
        metrics: PerformanceMetrics object
        
    Returns:
        Dictionary of metrics
    """
    return {
        'total_return': metrics.total_return,
        'annual_return': metrics.annual_return,
        'cumulative_return': metrics.cumulative_return,
        'volatility': metrics.volatility,
        'max_drawdown': metrics.max_drawdown,
        'avg_drawdown': metrics.avg_drawdown,
        'max_drawdown_duration': metrics.max_drawdown_duration,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'calmar_ratio': metrics.calmar_ratio,
        'total_trades': metrics.total_trades,
        'winning_trades': metrics.winning_trades,
        'losing_trades': metrics.losing_trades,
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'avg_win': metrics.avg_win,
        'avg_loss': metrics.avg_loss,
        'avg_trade': metrics.avg_trade,
        'largest_win': metrics.largest_win,
        'largest_loss': metrics.largest_loss,
        'exposure_time': metrics.exposure_time,
        'recovery_factor': metrics.recovery_factor,
        'expectancy': metrics.expectancy,
    }


def compare_metrics(metrics_list: List[PerformanceMetrics], 
                   labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple sets of metrics.
    
    Args:
        metrics_list: List of PerformanceMetrics objects
        labels: Optional labels for each metrics set
        
    Returns:
        DataFrame comparing metrics
    """
    if labels is None:
        labels = [f"Strategy {i+1}" for i in range(len(metrics_list))]
    
    data = []
    for metrics in metrics_list:
        data.append(metrics_to_dict(metrics))
    
    df = pd.DataFrame(data, index=labels).T
    return df
