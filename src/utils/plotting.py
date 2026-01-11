"""
Visualization Utilities for Strategy Analysis

Provides plotting functions for strategy performance, walk-forward analysis,
and parameter sensitivity.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve",
                     benchmark: Optional[pd.Series] = None) -> go.Figure:
    """
    Plot equity curve over time.
    
    Args:
        equity: Equity series
        title: Plot title
        benchmark: Optional benchmark equity series
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity.index if hasattr(equity, 'index') else list(range(len(equity))),
        y=equity,
        mode='lines',
        name='Strategy',
        line=dict(color='blue', width=2)
    ))
    
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index if hasattr(benchmark, 'index') else list(range(len(benchmark))),
            y=benchmark,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Equity",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> go.Figure:
    """
    Plot drawdown over time.
    
    Args:
        equity: Equity series
        title: Plot title
        
    Returns:
        Plotly figure
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index if hasattr(drawdown, 'index') else list(range(len(drawdown))),
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def plot_strategy_performance(df: pd.DataFrame, trades_df: pd.DataFrame,
                              equity: pd.Series, 
                              title: str = "Strategy Performance") -> go.Figure:
    """
    Create comprehensive strategy performance plot.
    
    Shows price, signals, equity curve, and drawdown.
    
    Args:
        df: OHLCV DataFrame with signals
        trades_df: DataFrame with trade information
        equity: Equity curve
        title: Plot title
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Equity Curve', 'Drawdown'),
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # Plot 1: Price and signals
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add entry/exit markers
    if not trades_df.empty:
        # Long entries
        long_entries = trades_df[trades_df['direction'] == 1]
        fig.add_trace(
            go.Scatter(
                x=long_entries['entry_time'],
                y=long_entries['entry_price'],
                mode='markers',
                name='Long Entry',
                marker=dict(symbol='triangle-up', size=10, color='green')
            ),
            row=1, col=1
        )
        
        # Short entries
        short_entries = trades_df[trades_df['direction'] == -1]
        fig.add_trace(
            go.Scatter(
                x=short_entries['entry_time'],
                y=short_entries['entry_price'],
                mode='markers',
                name='Short Entry',
                marker=dict(symbol='triangle-down', size=10, color='red')
            ),
            row=1, col=1
        )
        
        # Exits
        fig.add_trace(
            go.Scatter(
                x=trades_df['exit_time'],
                y=trades_df['exit_price'],
                mode='markers',
                name='Exit',
                marker=dict(symbol='x', size=8, color='black')
            ),
            row=1, col=1
        )
    
    # Plot 2: Equity curve
    equity_index = equity.index if hasattr(equity, 'index') else list(range(len(equity)))
    fig.add_trace(
        go.Scatter(
            x=equity_index,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Plot 3: Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=equity_index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis3_title="Time",
        yaxis1_title="Price",
        yaxis2_title="Equity",
        yaxis3_title="Drawdown (%)",
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update x-axis to show rangeslider
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def plot_walk_forward_efficiency(periods: List, 
                                 title: str = "Walk-Forward Analysis") -> go.Figure:
    """
    Plot walk-forward optimization results.
    
    Shows in-sample vs out-of-sample performance across periods.
    
    Args:
        periods: List of WalkForwardPeriod objects
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Extract data
    period_labels = [f"{p.test_start.strftime('%Y-%m')}" for p in periods]
    is_returns = [p.train_metrics.total_return * 100 if p.train_metrics else 0 for p in periods]
    oos_returns = [p.test_metrics.total_return * 100 if p.test_metrics else 0 for p in periods]
    is_sharpe = [p.train_metrics.sharpe_ratio if p.train_metrics else 0 for p in periods]
    oos_sharpe = [p.test_metrics.sharpe_ratio if p.test_metrics else 0 for p in periods]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Returns (%)', 'Sharpe Ratio')
    )
    
    # Plot returns
    fig.add_trace(
        go.Bar(
            x=period_labels,
            y=is_returns,
            name='In-Sample',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=period_labels,
            y=oos_returns,
            name='Out-of-Sample',
            marker_color='darkblue'
        ),
        row=1, col=1
    )
    
    # Plot Sharpe ratios
    fig.add_trace(
        go.Bar(
            x=period_labels,
            y=is_sharpe,
            name='In-Sample',
            marker_color='lightgreen',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=period_labels,
            y=oos_sharpe,
            name='Out-of-Sample',
            marker_color='darkgreen',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis2_title="Period",
        yaxis1_title="Return (%)",
        yaxis2_title="Sharpe Ratio",
        barmode='group',
        height=700,
        template='plotly_white'
    )
    
    return fig


def plot_parameter_stability(params_df: pd.DataFrame,
                            title: str = "Parameter Stability") -> go.Figure:
    """
    Plot parameter evolution across walk-forward periods.
    
    Args:
        params_df: DataFrame with parameters across periods
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for col in params_df.columns:
        fig.add_trace(go.Scatter(
            y=params_df[col],
            mode='lines+markers',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Parameter Value",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_trade_analysis(trades_df: pd.DataFrame,
                       title: str = "Trade Analysis") -> go.Figure:
    """
    Create trade analysis visualizations.
    
    Args:
        trades_df: DataFrame with trade information
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PnL Distribution', 'PnL Over Time', 
                       'Trade Duration', 'Cumulative PnL')
    )
    
    # PnL distribution
    fig.add_trace(
        go.Histogram(
            x=trades_df['pnl_pct'],
            name='PnL %',
            nbinsx=30,
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # PnL over time
    fig.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['pnl_pct'],
            mode='markers',
            name='Trade PnL',
            marker=dict(
                size=8,
                color=trades_df['pnl_pct'],
                colorscale='RdYlGn',
                showscale=True
            )
        ),
        row=1, col=2
    )
    
    # Trade duration
    if 'duration' in trades_df.columns:
        fig.add_trace(
            go.Histogram(
                x=trades_df['duration'],
                name='Duration (hours)',
                nbinsx=30,
                marker_color='green'
            ),
            row=2, col=1
        )
    
    # Cumulative PnL
    cumulative_pnl = trades_df['pnl'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def plot_metrics_heatmap(metrics_matrix: pd.DataFrame,
                         title: str = "Metrics Comparison") -> go.Figure:
    """
    Create heatmap of metrics comparison.
    
    Args:
        metrics_matrix: DataFrame with metrics across different runs
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=metrics_matrix.values,
        x=metrics_matrix.columns,
        y=metrics_matrix.index,
        colorscale='RdYlGn',
        text=metrics_matrix.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Strategy/Period",
        yaxis_title="Metric",
        height=600,
        template='plotly_white'
    )
    
    return fig


def plot_signal_distribution(df: pd.DataFrame,
                             title: str = "Signal Distribution") -> go.Figure:
    """
    Plot distribution of signals and scores.
    
    Args:
        df: DataFrame with signal columns
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Signal Distribution', 'Score Distribution')
    )
    
    # Signal distribution
    signal_counts = df['signal'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            name='Signals',
            marker_color=['red', 'gray', 'green']
        ),
        row=1, col=1
    )
    
    # Score distribution
    if 'adjusted_score' in df.columns:
        fig.add_trace(
            go.Histogram(
                x=df['adjusted_score'],
                name='Scores',
                nbinsx=50,
                marker_color='blue'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def save_figure(fig: go.Figure, filepath: str, 
               format: str = 'html') -> None:
    """
    Save plotly figure to file.
    
    Args:
        fig: Plotly figure
        filepath: Output file path
        format: Output format ('html', 'png', 'svg', 'pdf')
    """
    if format == 'html':
        fig.write_html(filepath)
    else:
        fig.write_image(filepath, format=format)
    
    logger.info(f"Figure saved to {filepath}")
