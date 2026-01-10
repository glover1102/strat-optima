"""
Performance Metrics Module

Calculates comprehensive performance metrics for backtesting and optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    expectancy: float
    recovery_factor: float
    ulcer_index: float


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate returns from equity curve
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Series of returns
    """
    return equity_curve.pct_change().fillna(0)


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Calculate total return"""
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def calculate_annual_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return
    
    Args:
        equity_curve: Series of equity values
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return percentage
    """
    if len(equity_curve) < 2:
        return 0.0
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    
    if years <= 0:
        return 0.0
    
    annual_return = (total_return ** (1 / years) - 1) * 100
    return annual_return


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / returns.std())
    
    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility)
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
    
    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_duration)
    """
    if len(equity_curve) < 2:
        return 0.0, 0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max * 100
    
    max_dd = abs(drawdown.min())
    
    # Calculate drawdown duration
    is_drawdown = drawdown < 0
    drawdown_periods = is_drawdown.astype(int).groupby(
        (is_drawdown != is_drawdown.shift()).cumsum()
    ).sum()
    
    max_dd_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
    
    return max_dd, max_dd_duration


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Args:
        annual_return: Annualized return
        max_drawdown: Maximum drawdown percentage
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    return annual_return / max_drawdown


def calculate_ulcer_index(equity_curve: pd.Series) -> float:
    """
    Calculate Ulcer Index (measure of downside volatility)
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Ulcer Index
    """
    if len(equity_curve) < 2:
        return 0.0
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    
    ulcer = np.sqrt(np.mean(drawdown ** 2))
    
    return abs(ulcer)


def calculate_trade_metrics(trades: pd.DataFrame) -> Dict:
    """
    Calculate trade-based metrics
    
    Args:
        trades: DataFrame with trade data (must have 'pnl', 'pnl_percent' columns)
        
    Returns:
        Dictionary with trade metrics
    """
    if len(trades) == 0:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'expectancy': 0.0
        }
    
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
    
    total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'expectancy': expectancy
    }


def calculate_all_metrics(equity_curve: pd.Series,
                         trades: Optional[pd.DataFrame] = None,
                         periods_per_year: int = 252,
                         risk_free_rate: float = 0.0) -> PerformanceMetrics:
    """
    Calculate all performance metrics
    
    Args:
        equity_curve: Series of equity values
        trades: DataFrame with trade data
        periods_per_year: Number of periods per year
        risk_free_rate: Annual risk-free rate
        
    Returns:
        PerformanceMetrics object
    """
    returns = calculate_returns(equity_curve)
    
    total_return = calculate_total_return(equity_curve)
    annual_return = calculate_annual_return(equity_curve, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(annual_return, max_dd)
    ulcer = calculate_ulcer_index(equity_curve)
    
    # Trade metrics
    if trades is not None and len(trades) > 0:
        trade_metrics = calculate_trade_metrics(trades)
        
        # Average trade duration
        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            trades['duration'] = (trades['exit_date'] - trades['entry_date']).dt.days
            avg_duration = trades['duration'].mean()
        else:
            avg_duration = 0.0
        
        # Recovery factor
        recovery_factor = abs(total_return / max_dd) if max_dd > 0 else 0.0
    else:
        trade_metrics = calculate_trade_metrics(pd.DataFrame())
        avg_duration = 0.0
        recovery_factor = 0.0
    
    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        win_rate=trade_metrics['win_rate'],
        profit_factor=trade_metrics['profit_factor'],
        avg_win=trade_metrics['avg_win'],
        avg_loss=trade_metrics['avg_loss'],
        total_trades=trade_metrics['total_trades'],
        winning_trades=trade_metrics['winning_trades'],
        losing_trades=trade_metrics['losing_trades'],
        avg_trade_duration=avg_duration,
        expectancy=trade_metrics['expectancy'],
        recovery_factor=recovery_factor,
        ulcer_index=ulcer
    )


def calculate_walk_forward_efficiency(in_sample_return: float,
                                      out_sample_return: float) -> float:
    """
    Calculate Walk-Forward Efficiency (WFE)
    
    WFE = Out-of-sample return / In-sample return
    
    Args:
        in_sample_return: In-sample return percentage
        out_sample_return: Out-of-sample return percentage
        
    Returns:
        Walk-forward efficiency ratio
    """
    if in_sample_return == 0:
        return 0.0
    
    return out_sample_return / in_sample_return


def compare_is_oos_metrics(is_metrics: PerformanceMetrics,
                           oos_metrics: PerformanceMetrics) -> Dict:
    """
    Compare in-sample and out-of-sample metrics
    
    Args:
        is_metrics: In-sample performance metrics
        oos_metrics: Out-of-sample performance metrics
        
    Returns:
        Dictionary with comparison metrics
    """
    wfe = calculate_walk_forward_efficiency(
        is_metrics.total_return,
        oos_metrics.total_return
    )
    
    return {
        'wfe': wfe,
        'return_degradation': is_metrics.total_return - oos_metrics.total_return,
        'sharpe_degradation': is_metrics.sharpe_ratio - oos_metrics.sharpe_ratio,
        'dd_increase': oos_metrics.max_drawdown - is_metrics.max_drawdown,
        'win_rate_change': oos_metrics.win_rate - is_metrics.win_rate,
        'is_sharpe': is_metrics.sharpe_ratio,
        'oos_sharpe': oos_metrics.sharpe_ratio,
        'is_return': is_metrics.total_return,
        'oos_return': oos_metrics.total_return
    }


def metrics_to_dict(metrics: PerformanceMetrics) -> Dict:
    """Convert PerformanceMetrics to dictionary"""
    return {
        'total_return': metrics.total_return,
        'annual_return': metrics.annual_return,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'calmar_ratio': metrics.calmar_ratio,
        'max_drawdown': metrics.max_drawdown,
        'max_drawdown_duration': metrics.max_drawdown_duration,
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'avg_win': metrics.avg_win,
        'avg_loss': metrics.avg_loss,
        'total_trades': metrics.total_trades,
        'winning_trades': metrics.winning_trades,
        'losing_trades': metrics.losing_trades,
        'avg_trade_duration': metrics.avg_trade_duration,
        'expectancy': metrics.expectancy,
        'recovery_factor': metrics.recovery_factor,
        'ulcer_index': metrics.ulcer_index
    }
