"""
QTAlgo Super26 Strategy - Signal Generation Module

This module implements the signal generation logic with dynamic scoring system:
- Combines all 7 indicators with configurable weights
- ADX-based penalty system for weak trends
- Dynamic minimum score thresholds
- Signal strength classification
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_indicator_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate individual indicator scores
    
    Args:
        df: DataFrame with all indicators calculated
        weights: Dictionary of indicator weights
        
    Returns:
        DataFrame with individual indicator scores
    """
    result = df.copy()
    
    # ADX Score
    # Score based on trend strength and directional indicators
    adx_score = np.where(
        (result['adx'] >= 25) & (result['plus_di'] > result['minus_di']), weights['w_adx'],
        np.where(
            (result['adx'] >= 25) & (result['plus_di'] < result['minus_di']), -weights['w_adx'],
            np.where(
                (result['adx'] >= 20) & (result['plus_di'] > result['minus_di']), weights['w_adx'] * 0.5,
                np.where(
                    (result['adx'] >= 20) & (result['plus_di'] < result['minus_di']), -weights['w_adx'] * 0.5,
                    0
                )
            )
        )
    )
    
    # Regime Filter Score
    regime_score = result['regime_signal'] * weights['w_regime']
    
    # Pivot Trend Score (primary signal)
    pivot_score = result['pivot_trend'] * weights['w_pivotTrend']
    
    # Trend Duration Score
    trend_duration_score = result['trend_signal'] * weights['w_trendDuration']
    
    # ML SuperTrend Score
    ml_supertrend_score = result['ml_supertrend'] * weights['w_mlSupertrend']
    
    # Linear Regression Channel Score
    linreg_score = result['linreg_signal'] * weights['w_linregChannel']
    
    # Pivot Levels Score
    pivot_levels_score = result['pivot_signal'] * weights['w_pivotLevels']
    
    result['score_adx'] = adx_score
    result['score_regime'] = regime_score
    result['score_pivot_trend'] = pivot_score
    result['score_trend_duration'] = trend_duration_score
    result['score_ml_supertrend'] = ml_supertrend_score
    result['score_linreg'] = linreg_score
    result['score_pivot_levels'] = pivot_levels_score
    
    return result


def calculate_total_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total composite score from all indicators
    
    Args:
        df: DataFrame with individual indicator scores
        
    Returns:
        DataFrame with total score
    """
    result = df.copy()
    
    score_columns = [
        'score_adx',
        'score_regime',
        'score_pivot_trend',
        'score_trend_duration',
        'score_ml_supertrend',
        'score_linreg',
        'score_pivot_levels'
    ]
    
    # Sum all indicator scores
    result['total_score'] = result[score_columns].sum(axis=1)
    
    # Separate long and short scores
    result['long_score'] = result['total_score'].clip(lower=0)
    result['short_score'] = abs(result['total_score'].clip(upper=0))
    
    return result


def apply_adx_penalty(df: pd.DataFrame, adx_threshold: float = 20) -> pd.DataFrame:
    """
    Apply ADX-based penalty for weak trends
    
    Args:
        df: DataFrame with scores and ADX
        adx_threshold: ADX threshold for weak trend detection
        
    Returns:
        DataFrame with adjusted scores
    """
    result = df.copy()
    
    # Penalty factor based on ADX
    # If ADX < threshold, reduce the score
    penalty_factor = np.where(
        result['adx'] < adx_threshold,
        result['adx'] / adx_threshold,  # Scale down proportionally
        1.0  # No penalty
    )
    
    result['penalty_factor'] = penalty_factor
    result['adjusted_long_score'] = result['long_score'] * penalty_factor
    result['adjusted_short_score'] = result['short_score'] * penalty_factor
    
    return result


def generate_entry_signals(df: pd.DataFrame, 
                          strong_trend_min_score: float = 1.5,
                          weak_trend_min_score: float = 3.0,
                          adx_threshold: float = 20) -> pd.DataFrame:
    """
    Generate entry signals based on dynamic scoring
    
    Args:
        df: DataFrame with adjusted scores
        strong_trend_min_score: Minimum score for strong trend entries
        weak_trend_min_score: Minimum score for weak trend entries
        adx_threshold: ADX threshold for trend strength
        
    Returns:
        DataFrame with entry signals
    """
    result = df.copy()
    
    # Determine if we're in a strong or weak trend
    is_strong_trend = result['adx'] >= adx_threshold
    
    # Dynamic threshold based on trend strength
    long_threshold = np.where(is_strong_trend, strong_trend_min_score, weak_trend_min_score)
    short_threshold = np.where(is_strong_trend, strong_trend_min_score, weak_trend_min_score)
    
    # Generate signals
    long_signal = (result['adjusted_long_score'] >= long_threshold) & (result['adjusted_long_score'] > result['adjusted_short_score'])
    short_signal = (result['adjusted_short_score'] >= short_threshold) & (result['adjusted_short_score'] > result['adjusted_long_score'])
    
    # Signal values: 1 = long, -1 = short, 0 = no signal
    result['entry_signal'] = np.where(long_signal, 1,
                                      np.where(short_signal, -1, 0))
    
    # Signal strength classification
    max_score = np.maximum(result['adjusted_long_score'], result['adjusted_short_score'])
    result['signal_strength'] = np.where(
        max_score >= weak_trend_min_score * 1.5, 'very_strong',
        np.where(
            max_score >= weak_trend_min_score, 'strong',
            np.where(
                max_score >= strong_trend_min_score, 'moderate',
                'weak'
            )
        )
    )
    
    return result


def detect_signal_reversal(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detect signal reversals and weakening
    
    Args:
        df: DataFrame with entry signals
        lookback: Number of bars to look back for reversal detection
        
    Returns:
        DataFrame with reversal signals
    """
    result = df.copy()
    
    # Signal change detection
    signal_changed = result['entry_signal'] != result['entry_signal'].shift(1)
    
    # Detect score weakening
    current_score = np.maximum(result['adjusted_long_score'], result['adjusted_short_score'])
    avg_score = current_score.rolling(window=lookback).mean()
    
    score_weakening = current_score < (avg_score * 0.7)  # 30% decline from average
    
    # Reversal signal
    result['signal_reversal'] = signal_changed
    result['score_weakening'] = score_weakening
    
    return result


def generate_all_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate all trading signals for the Super26 strategy
    
    Args:
        df: DataFrame with all indicators calculated
        params: Dictionary of strategy parameters
        
    Returns:
        DataFrame with all signals
    """
    # Extract parameters
    weights = {
        'w_adx': params.get('w_adx', 1.0),
        'w_regime': params.get('w_regime', 1.0),
        'w_pivotTrend': params.get('w_pivotTrend', 1.5),
        'w_trendDuration': params.get('w_trendDuration', 0.8),
        'w_mlSupertrend': params.get('w_mlSupertrend', 1.2),
        'w_linregChannel': params.get('w_linregChannel', 0.9),
        'w_pivotLevels': params.get('w_pivotLevels', 0.7)
    }
    
    strong_trend_min_score = params.get('strongTrendMinScore', 1.5)
    weak_trend_min_score = params.get('weakTrendMinScore', 3.0)
    adx_threshold = params.get('adxThreshold', 20)
    
    # Calculate scores
    result = calculate_indicator_scores(df, weights)
    result = calculate_total_score(result)
    result = apply_adx_penalty(result, adx_threshold)
    
    # Generate signals
    result = generate_entry_signals(result, 
                                   strong_trend_min_score,
                                   weak_trend_min_score,
                                   adx_threshold)
    
    # Detect reversals
    result = detect_signal_reversal(result)
    
    return result


def get_signal_details(df: pd.DataFrame, index: int) -> Dict:
    """
    Get detailed signal information for a specific bar
    
    Args:
        df: DataFrame with signals
        index: Index of the bar
        
    Returns:
        Dictionary with signal details
    """
    if index < 0 or index >= len(df):
        return {}
    
    row = df.iloc[index]
    
    details = {
        'entry_signal': row['entry_signal'],
        'signal_strength': row['signal_strength'],
        'total_score': row['total_score'],
        'long_score': row['long_score'],
        'short_score': row['short_score'],
        'adjusted_long_score': row['adjusted_long_score'],
        'adjusted_short_score': row['adjusted_short_score'],
        'penalty_factor': row['penalty_factor'],
        'adx': row['adx'],
        'individual_scores': {
            'adx': row['score_adx'],
            'regime': row['score_regime'],
            'pivot_trend': row['score_pivot_trend'],
            'trend_duration': row['score_trend_duration'],
            'ml_supertrend': row['score_ml_supertrend'],
            'linreg': row['score_linreg'],
            'pivot_levels': row['score_pivot_levels']
        },
        'reversal_signals': {
            'signal_reversal': row['signal_reversal'],
            'score_weakening': row['score_weakening']
        }
    }
    
    return details


def filter_signals(df: pd.DataFrame, 
                  min_bars_between_signals: int = 5,
                  require_all_indicators: bool = False) -> pd.DataFrame:
    """
    Filter signals based on additional criteria
    
    Args:
        df: DataFrame with signals
        min_bars_between_signals: Minimum bars between consecutive signals
        require_all_indicators: If True, require all indicators to agree
        
    Returns:
        DataFrame with filtered signals
    """
    result = df.copy()
    
    # Filter by minimum bars between signals
    if min_bars_between_signals > 0:
        filtered_signal = result['entry_signal'].copy()
        last_signal_idx = -min_bars_between_signals
        
        for i in range(len(result)):
            if result['entry_signal'].iloc[i] != 0:
                if i - last_signal_idx < min_bars_between_signals:
                    filtered_signal.iloc[i] = 0
                else:
                    last_signal_idx = i
        
        result['filtered_entry_signal'] = filtered_signal
    else:
        result['filtered_entry_signal'] = result['entry_signal']
    
    # Require all indicators to agree (unanimous)
    if require_all_indicators:
        score_columns = ['score_adx', 'score_regime', 'score_pivot_trend', 
                        'score_trend_duration', 'score_ml_supertrend', 
                        'score_linreg', 'score_pivot_levels']
        
        all_bullish = (result[score_columns] > 0).all(axis=1)
        all_bearish = (result[score_columns] < 0).all(axis=1)
        
        result['filtered_entry_signal'] = np.where(
            all_bullish & (result['filtered_entry_signal'] == 1), 1,
            np.where(all_bearish & (result['filtered_entry_signal'] == -1), -1, 0)
        )
    
    return result
