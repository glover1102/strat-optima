"""
Plotting and Visualization Module

Creates comprehensive visualizations for strategy performance analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path

from ..optimization.metrics import PerformanceMetrics
from ..optimization.walk_forward import WalkForwardResults


def plot_equity_curve(equity: pd.Series,
                     title: str = "Equity Curve",
                     show_drawdown: bool = True,
                     save_path: Optional[str] = None) -> go.Figure:
    """
    Plot equity curve with optional drawdown
    
    Args:
        equity: Equity curve series
        title: Plot title
        show_drawdown: Whether to show drawdown subplot
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, "Drawdown %"),
            vertical_spacing=0.1
        )
    else:
        fig = go.Figure()
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    if show_drawdown:
        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Equity ($)")
    
    fig.update_xaxes(title_text="Date")
    fig.update_layout(height=600, showlegend=True)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_parameter_evolution(param_history: pd.DataFrame,
                            save_path: Optional[str] = None) -> go.Figure:
    """
    Plot parameter evolution across walk-forward windows
    
    Args:
        param_history: DataFrame with parameter values per window
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    param_cols = [col for col in param_history.columns if col != 'window_id']
    
    n_params = len(param_cols)
    rows = (n_params + 2) // 3
    
    fig = make_subplots(
        rows=rows, cols=3,
        subplot_titles=param_cols
    )
    
    for idx, param in enumerate(param_cols):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Scatter(
                x=param_history['window_id'] if 'window_id' in param_history.columns else list(range(len(param_history))),
                y=param_history[param],
                mode='lines+markers',
                name=param,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Window", row=row, col=col)
        fig.update_yaxes(title_text=param, row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title_text="Parameter Evolution Across Windows"
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_walk_forward_efficiency(results: WalkForwardResults,
                                 save_path: Optional[str] = None) -> go.Figure:
    """
    Plot walk-forward efficiency metrics
    
    Args:
        results: WalkForwardResults object
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    # Calculate WFE for each window
    window_ids = []
    is_returns = []
    oos_returns = []
    wfes = []
    
    for window in results.windows:
        window_ids.append(window.window_id)
        is_returns.append(window.is_metrics.total_return)
        oos_returns.append(window.oos_metrics.total_return)
        
        if window.is_metrics.total_return != 0:
            wfe = window.oos_metrics.total_return / window.is_metrics.total_return
        else:
            wfe = 0
        wfes.append(wfe)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("In-Sample vs Out-of-Sample Returns", "Walk-Forward Efficiency"),
        vertical_spacing=0.15
    )
    
    # Returns comparison
    fig.add_trace(
        go.Bar(
            x=window_ids,
            y=is_returns,
            name='In-Sample',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=window_ids,
            y=oos_returns,
            name='Out-of-Sample',
            marker_color='darkblue'
        ),
        row=1, col=1
    )
    
    # WFE
    fig.add_trace(
        go.Scatter(
            x=window_ids,
            y=wfes,
            mode='lines+markers',
            name='WFE',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Add horizontal line at WFE = 1
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_xaxes(title_text="Window", row=1, col=1)
    fig.update_xaxes(title_text="Window", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="WFE Ratio", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_performance_metrics(metrics: PerformanceMetrics,
                            title: str = "Performance Metrics",
                            save_path: Optional[str] = None) -> go.Figure:
    """
    Plot performance metrics as bar chart
    
    Args:
        metrics: PerformanceMetrics object
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    metric_names = [
        'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
        'Max Drawdown %', 'Win Rate %', 'Profit Factor'
    ]
    
    metric_values = [
        metrics.sharpe_ratio,
        metrics.sortino_ratio,
        metrics.calmar_ratio,
        metrics.max_drawdown,
        metrics.win_rate * 100,
        metrics.profit_factor
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['green' if v > 0 else 'red' for v in metric_values]
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_signal_distribution(df: pd.DataFrame,
                            save_path: Optional[str] = None) -> go.Figure:
    """
    Plot distribution of entry signals
    
    Args:
        df: DataFrame with signals
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    if 'entry_signal' not in df.columns:
        raise ValueError("DataFrame must have 'entry_signal' column")
    
    signal_counts = df['entry_signal'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Long', 'Neutral', 'Short'],
            values=[
                signal_counts.get(1, 0),
                signal_counts.get(0, 0),
                signal_counts.get(-1, 0)
            ],
            marker=dict(colors=['green', 'gray', 'red'])
        )
    ])
    
    fig.update_layout(
        title="Signal Distribution",
        height=400
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_parameter_stability(stability: Dict[str, float],
                            save_path: Optional[str] = None) -> go.Figure:
    """
    Plot parameter stability scores
    
    Args:
        stability: Dictionary of parameter stability scores
        save_path: Path to save the plot
        
    Returns:
        Plotly figure
    """
    params = list(stability.keys())
    scores = list(stability.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=params,
            y=scores,
            marker_color=['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in scores]
        )
    ])
    
    fig.update_layout(
        title="Parameter Stability (Higher is Better)",
        xaxis_title="Parameter",
        yaxis_title="Stability Score",
        height=400
    )
    
    # Add stability threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Fair")
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def create_dashboard(results: WalkForwardResults,
                    equity: pd.Series,
                    param_history: pd.DataFrame,
                    output_dir: str):
    """
    Create comprehensive dashboard with all plots
    
    Args:
        results: WalkForwardResults object
        equity: Combined OOS equity curve
        param_history: Parameter evolution DataFrame
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Equity curve
    plot_equity_curve(
        equity,
        title="Combined Out-of-Sample Equity Curve",
        save_path=str(output_path / "equity_curve.html")
    )
    
    # Parameter evolution
    plot_parameter_evolution(
        param_history,
        save_path=str(output_path / "parameter_evolution.html")
    )
    
    # Walk-forward efficiency
    plot_walk_forward_efficiency(
        results,
        save_path=str(output_path / "walk_forward_efficiency.html")
    )
    
    # Performance metrics
    plot_performance_metrics(
        results.combined_oos_metrics,
        title="Combined Out-of-Sample Performance",
        save_path=str(output_path / "performance_metrics.html")
    )
    
    # Parameter stability
    plot_parameter_stability(
        results.parameter_stability,
        save_path=str(output_path / "parameter_stability.html")
    )
    
    print(f"Dashboard created in {output_dir}")
