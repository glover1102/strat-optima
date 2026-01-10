"""
Reporting Module

Generates comprehensive reports for optimization results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

from ..optimization.metrics import PerformanceMetrics, metrics_to_dict
from ..optimization.walk_forward import WalkForwardResults


def generate_summary_report(results: WalkForwardResults) -> Dict:
    """
    Generate summary report from walk-forward results
    
    Args:
        results: WalkForwardResults object
        
    Returns:
        Dictionary with summary statistics
    """
    # Calculate aggregate statistics
    all_is_sharpes = [w.is_metrics.sharpe_ratio for w in results.windows]
    all_oos_sharpes = [w.oos_metrics.sharpe_ratio for w in results.windows]
    all_is_returns = [w.is_metrics.total_return for w in results.windows]
    all_oos_returns = [w.oos_metrics.total_return for w in results.windows]
    
    # Calculate WFEs
    wfes = []
    for window in results.windows:
        if window.is_metrics.total_return != 0:
            wfe = window.oos_metrics.total_return / window.is_metrics.total_return
            wfes.append(wfe)
    
    summary = {
        'optimization': {
            'total_windows': results.total_windows,
            'optimization_time': results.optimization_time,
            'window_type': 'rolling',  # Could be made configurable
        },
        'combined_oos_performance': metrics_to_dict(results.combined_oos_metrics),
        'window_statistics': {
            'avg_is_sharpe': np.mean(all_is_sharpes),
            'avg_oos_sharpe': np.mean(all_oos_sharpes),
            'avg_is_return': np.mean(all_is_returns),
            'avg_oos_return': np.mean(all_oos_returns),
            'sharpe_degradation': np.mean(all_is_sharpes) - np.mean(all_oos_sharpes),
            'return_degradation': np.mean(all_is_returns) - np.mean(all_oos_returns),
        },
        'walk_forward_efficiency': {
            'average_wfe': results.average_wfe,
            'median_wfe': np.median(wfes) if wfes else 0,
            'min_wfe': np.min(wfes) if wfes else 0,
            'max_wfe': np.max(wfes) if wfes else 0,
            'std_wfe': np.std(wfes) if wfes else 0,
        },
        'parameter_stability': results.parameter_stability,
        'generated_at': datetime.now().isoformat()
    }
    
    return summary


def generate_window_report(results: WalkForwardResults) -> pd.DataFrame:
    """
    Generate detailed report for each window
    
    Args:
        results: WalkForwardResults object
        
    Returns:
        DataFrame with window-by-window results
    """
    rows = []
    
    for window in results.windows:
        # Calculate WFE
        if window.is_metrics.total_return != 0:
            wfe = window.oos_metrics.total_return / window.is_metrics.total_return
        else:
            wfe = 0
        
        row = {
            'window_id': window.window_id,
            'train_start': window.train_start,
            'train_end': window.train_end,
            'test_start': window.test_start,
            'test_end': window.test_end,
            
            # In-sample metrics
            'is_return': window.is_metrics.total_return,
            'is_sharpe': window.is_metrics.sharpe_ratio,
            'is_max_dd': window.is_metrics.max_drawdown,
            'is_win_rate': window.is_metrics.win_rate,
            'is_trades': window.is_metrics.total_trades,
            
            # Out-of-sample metrics
            'oos_return': window.oos_metrics.total_return,
            'oos_sharpe': window.oos_metrics.sharpe_ratio,
            'oos_max_dd': window.oos_metrics.max_drawdown,
            'oos_win_rate': window.oos_metrics.win_rate,
            'oos_trades': window.oos_metrics.total_trades,
            
            # Comparison
            'wfe': wfe,
            'return_degradation': window.is_metrics.total_return - window.oos_metrics.total_return,
            'sharpe_degradation': window.is_metrics.sharpe_ratio - window.oos_metrics.sharpe_ratio,
        }
        
        # Add optimal parameters
        for param_name, param_value in window.optimal_params.items():
            row[f'param_{param_name}'] = param_value
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_parameter_analysis(results: WalkForwardResults) -> Dict:
    """
    Generate analysis of parameter behavior across windows
    
    Args:
        results: WalkForwardResults object
        
    Returns:
        Dictionary with parameter analysis
    """
    # Extract parameter history
    param_history = pd.DataFrame([w.optimal_params for w in results.windows])
    
    analysis = {}
    
    for param in param_history.columns:
        values = param_history[param]
        
        analysis[param] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'range': float(values.max() - values.min()),
            'cv': float(values.std() / values.mean()) if values.mean() != 0 else 0,
            'stability_score': results.parameter_stability.get(param, 0),
            
            # Trend detection
            'trend_slope': float(np.polyfit(range(len(values)), values, 1)[0]),
            'is_drifting': abs(np.polyfit(range(len(values)), values, 1)[0]) > 0.1,
        }
    
    return analysis


def generate_trade_analysis(results: WalkForwardResults) -> Dict:
    """
    Generate trade analysis across all windows
    
    Args:
        results: WalkForwardResults object
        
    Returns:
        Dictionary with trade analysis
    """
    all_trades = {
        'in_sample': [],
        'out_of_sample': []
    }
    
    for window in results.windows:
        all_trades['in_sample'].append({
            'window_id': window.window_id,
            'total_trades': window.is_metrics.total_trades,
            'win_rate': window.is_metrics.win_rate,
            'avg_win': window.is_metrics.avg_win,
            'avg_loss': window.is_metrics.avg_loss,
            'profit_factor': window.is_metrics.profit_factor,
        })
        
        all_trades['out_of_sample'].append({
            'window_id': window.window_id,
            'total_trades': window.oos_metrics.total_trades,
            'win_rate': window.oos_metrics.win_rate,
            'avg_win': window.oos_metrics.avg_win,
            'avg_loss': window.oos_metrics.avg_loss,
            'profit_factor': window.oos_metrics.profit_factor,
        })
    
    # Aggregate statistics
    is_df = pd.DataFrame(all_trades['in_sample'])
    oos_df = pd.DataFrame(all_trades['out_of_sample'])
    
    analysis = {
        'in_sample': {
            'avg_trades_per_window': is_df['total_trades'].mean(),
            'avg_win_rate': is_df['win_rate'].mean(),
            'avg_profit_factor': is_df['profit_factor'].mean(),
        },
        'out_of_sample': {
            'avg_trades_per_window': oos_df['total_trades'].mean(),
            'avg_win_rate': oos_df['win_rate'].mean(),
            'avg_profit_factor': oos_df['profit_factor'].mean(),
        },
        'degradation': {
            'win_rate_change': oos_df['win_rate'].mean() - is_df['win_rate'].mean(),
            'profit_factor_change': oos_df['profit_factor'].mean() - is_df['profit_factor'].mean(),
        }
    }
    
    return analysis


def generate_risk_analysis(results: WalkForwardResults) -> Dict:
    """
    Generate risk analysis
    
    Args:
        results: WalkForwardResults object
        
    Returns:
        Dictionary with risk metrics
    """
    oos_dds = [w.oos_metrics.max_drawdown for w in results.windows]
    oos_returns = [w.oos_metrics.total_return for w in results.windows]
    
    return {
        'max_drawdown': results.combined_oos_metrics.max_drawdown,
        'avg_window_drawdown': np.mean(oos_dds),
        'max_window_drawdown': np.max(oos_dds),
        'drawdown_volatility': np.std(oos_dds),
        'return_volatility': np.std(oos_returns),
        'ulcer_index': results.combined_oos_metrics.ulcer_index,
        'recovery_factor': results.combined_oos_metrics.recovery_factor,
    }


def export_full_report(results: WalkForwardResults,
                      output_dir: str,
                      format: str = 'json'):
    """
    Export comprehensive report to file
    
    Args:
        results: WalkForwardResults object
        output_dir: Output directory
        format: Export format ('json', 'csv', or 'both')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate all reports
    summary = generate_summary_report(results)
    window_df = generate_window_report(results)
    param_analysis = generate_parameter_analysis(results)
    trade_analysis = generate_trade_analysis(results)
    risk_analysis = generate_risk_analysis(results)
    
    # Combine into full report
    full_report = {
        'summary': summary,
        'parameter_analysis': param_analysis,
        'trade_analysis': trade_analysis,
        'risk_analysis': risk_analysis,
    }
    
    # Export JSON
    if format in ['json', 'both']:
        with open(output_path / 'full_report.json', 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
    
    # Export CSV
    if format in ['csv', 'both']:
        window_df.to_csv(output_path / 'window_results.csv', index=False)
        
        # Export parameter analysis as CSV
        param_df = pd.DataFrame(param_analysis).T
        param_df.to_csv(output_path / 'parameter_analysis.csv')
    
    print(f"Full report exported to {output_dir}")


def print_summary_report(results: WalkForwardResults):
    """
    Print formatted summary report to console
    
    Args:
        results: WalkForwardResults object
    """
    summary = generate_summary_report(results)
    
    print("\n" + "="*80)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"\nOptimization Info:")
    print(f"  Total Windows: {summary['optimization']['total_windows']}")
    print(f"  Optimization Time: {summary['optimization']['optimization_time']:.1f}s")
    
    print(f"\nCombined Out-of-Sample Performance:")
    oos = summary['combined_oos_performance']
    print(f"  Total Return: {oos['total_return']:.2f}%")
    print(f"  Annual Return: {oos['annual_return']:.2f}%")
    print(f"  Sharpe Ratio: {oos['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {oos['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {oos['max_drawdown']:.2f}%")
    print(f"  Win Rate: {oos['win_rate']*100:.1f}%")
    print(f"  Total Trades: {oos['total_trades']}")
    
    print(f"\nWalk-Forward Efficiency:")
    wfe = summary['walk_forward_efficiency']
    print(f"  Average WFE: {wfe['average_wfe']:.2f}")
    print(f"  Median WFE: {wfe['median_wfe']:.2f}")
    print(f"  WFE Range: [{wfe['min_wfe']:.2f}, {wfe['max_wfe']:.2f}]")
    
    print(f"\nParameter Stability:")
    for param, stability in sorted(summary['parameter_stability'].items(), 
                                   key=lambda x: x[1], reverse=True):
        status = "✓" if stability > 0.7 else "⚠" if stability > 0.5 else "✗"
        print(f"  {status} {param}: {stability:.3f}")
    
    print("\n" + "="*80 + "\n")
