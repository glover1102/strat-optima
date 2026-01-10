"""
Utility Module

Helper functions for plotting, reporting, and analysis.
"""

from .plotting import (
    plot_equity_curve,
    plot_parameter_evolution,
    plot_walk_forward_efficiency,
    plot_performance_metrics,
    plot_signal_distribution,
    plot_parameter_stability,
    create_dashboard
)

from .reporting import (
    generate_summary_report,
    generate_window_report,
    generate_parameter_analysis,
    generate_trade_analysis,
    generate_risk_analysis,
    export_full_report,
    print_summary_report
)

__all__ = [
    # Plotting
    'plot_equity_curve',
    'plot_parameter_evolution',
    'plot_walk_forward_efficiency',
    'plot_performance_metrics',
    'plot_signal_distribution',
    'plot_parameter_stability',
    'create_dashboard',
    # Reporting
    'generate_summary_report',
    'generate_window_report',
    'generate_parameter_analysis',
    'generate_trade_analysis',
    'generate_risk_analysis',
    'export_full_report',
    'print_summary_report'
]
