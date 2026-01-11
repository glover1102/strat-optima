"""
QTAlgo Super26 Strategy Module

This module contains the core strategy implementation including:
- Technical indicators
- Signal generation
- Exit management
"""

from .indicators import (
    calculate_adx,
    calculate_regime_filter,
    calculate_pivot_trend,
    calculate_trend_duration,
    calculate_ml_supertrend,
    calculate_linreg_channel,
    calculate_pivot_levels,
    calculate_all_indicators,
    hma
)

from .signals import (
    calculate_indicator_scores,
    calculate_total_score,
    apply_adx_penalty,
    generate_entry_signals,
    detect_signal_reversal,
    generate_all_signals,
    get_signal_details,
    filter_signals
)

from .exits import (
    Position,
    ExitManager,
    simulate_exits,
    calculate_exit_performance,
    analyze_exit_reasons,
    get_active_exit_levels
)

__all__ = [
    # Indicators
    'calculate_adx',
    'calculate_regime_filter',
    'calculate_pivot_trend',
    'calculate_trend_duration',
    'calculate_ml_supertrend',
    'calculate_linreg_channel',
    'calculate_pivot_levels',
    'calculate_all_indicators',
    'hma',
    # Signals
    'calculate_indicator_scores',
    'calculate_total_score',
    'apply_adx_penalty',
    'generate_entry_signals',
    'detect_signal_reversal',
    'generate_all_signals',
    'get_signal_details',
    'filter_signals',
    # Exits
    'Position',
    'ExitManager',
    'simulate_exits',
    'calculate_exit_performance',
    'analyze_exit_reasons',
    'get_active_exit_levels'
]
