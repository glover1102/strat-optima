"""
Optimization Module

Handles walk-forward optimization, parameter space management, and performance metrics.
"""

from .walk_forward import (
    WalkForwardOptimizer,
    WalkForwardWindow,
    WalkForwardResults
)

from .metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_walk_forward_efficiency,
    compare_is_oos_metrics,
    metrics_to_dict
)

from .parameter_space import (
    ParameterSpace,
    ParameterDefinition,
    create_parameter_space_from_config,
    calculate_parameter_stability,
    detect_parameter_drift,
    suggest_parameter_ranges
)

__all__ = [
    # Walk-forward
    'WalkForwardOptimizer',
    'WalkForwardWindow',
    'WalkForwardResults',
    # Metrics
    'PerformanceMetrics',
    'calculate_all_metrics',
    'calculate_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_walk_forward_efficiency',
    'compare_is_oos_metrics',
    'metrics_to_dict',
    # Parameter space
    'ParameterSpace',
    'ParameterDefinition',
    'create_parameter_space_from_config',
    'calculate_parameter_stability',
    'detect_parameter_drift',
    'suggest_parameter_ranges'
]
