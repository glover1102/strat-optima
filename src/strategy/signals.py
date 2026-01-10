"""
Signal Generation Logic for QTAlgo Super26 Strategy

Implements the dynamic scoring system that combines all indicators
with configurable weights and ADX-based penalty system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class SignalGenerator:
    """
    Generates trading signals based on the QTAlgo Super26 strategy.
    
    Combines scores from all 7 indicators with configurable weights
    and applies ADX-based penalties for weak trends.
    """
    
    def __init__(self, params: Dict):
        """
        Initialize signal generator with parameters.
        
        Args:
            params: Dictionary containing strategy parameters including:
                - strongTrendMinScore: Minimum score for strong trend entries
                - weakTrendMinScore: Minimum score for weak trend entries
                - adx_threshold: ADX threshold for trend classification
                - Indicator weights (w_adx, w_regime, etc.)
        """
        self.params = params
        
        # Entry thresholds
        self.strong_trend_min = params.get('strongTrendMinScore', 1.5)
        self.weak_trend_min = params.get('weakTrendMinScore', 3.0)
        self.adx_threshold = params.get('adx_threshold', 20)
        
        # Indicator weights
        self.weights = {
            'adx': params.get('w_adx', 1.0),
            'regime': params.get('w_regime', 1.0),
            'pivot_trend': params.get('w_pivotTrend', 1.0),
            'trend_duration': params.get('w_trendDuration', 1.0),
            'ml_supertrend': params.get('w_mlSupertrend', 1.0),
            'linreg': params.get('w_linReg', 1.0),
            'pivot_levels': params.get('w_pivotLevels', 1.0),
        }
    
    def calculate_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite score from all indicators.
        
        Args:
            df: DataFrame with all indicator scores
            
        Returns:
            Series with composite scores
        """
        score = (
            self.weights['adx'] * df['adx_score'] +
            self.weights['regime'] * df['regime_score'] +
            self.weights['pivot_trend'] * df['pivot_trend_score'] +
            self.weights['trend_duration'] * df['trend_duration_score'] +
            self.weights['ml_supertrend'] * df['ml_supertrend_score'] +
            self.weights['linreg'] * df['linreg_score'] +
            self.weights['pivot_levels'] * df['pivot_levels_score']
        )
        
        return score
    
    def apply_adx_penalty(self, score: pd.Series, adx: pd.Series,
                         is_strong_trend: pd.Series) -> pd.Series:
        """
        Apply ADX-based penalty for weak trends.
        
        In weak trend conditions (ADX < threshold), signals require
        higher confirmation scores.
        
        Args:
            score: Composite score series
            adx: ADX values
            is_strong_trend: Boolean series indicating strong trends
            
        Returns:
            Adjusted score series
        """
        # In weak trends, double the required score threshold
        penalty = np.where(~is_strong_trend, 0.5, 1.0)
        adjusted_score = score * penalty
        
        return adjusted_score
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry and exit signals based on indicator scores.
        
        Args:
            df: DataFrame with all indicator columns
            
        Returns:
            DataFrame with signal columns added:
                - composite_score: Raw composite score
                - is_strong_trend: Boolean for strong trend
                - adjusted_score: Score after ADX penalty
                - signal: 1 for long, -1 for short, 0 for no signal
                - signal_strength: Absolute value of adjusted score
        """
        result = df.copy()
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(result)
        result['composite_score'] = composite_score
        
        # Determine if strong trend
        is_strong_trend = result['adx'] >= self.adx_threshold
        result['is_strong_trend'] = is_strong_trend
        
        # Apply ADX penalty
        adjusted_score = self.apply_adx_penalty(
            composite_score, 
            result['adx'],
            is_strong_trend
        )
        result['adjusted_score'] = adjusted_score
        
        # Generate signals based on thresholds
        signal = np.zeros(len(result))
        
        # Long signals
        long_strong = (is_strong_trend & 
                      (adjusted_score >= self.strong_trend_min))
        long_weak = (~is_strong_trend & 
                    (adjusted_score >= self.weak_trend_min))
        signal = np.where(long_strong | long_weak, 1, signal)
        
        # Short signals
        short_strong = (is_strong_trend & 
                       (adjusted_score <= -self.strong_trend_min))
        short_weak = (~is_strong_trend & 
                     (adjusted_score <= -self.weak_trend_min))
        signal = np.where(short_strong | short_weak, -1, signal)
        
        result['signal'] = signal
        result['signal_strength'] = np.abs(adjusted_score)
        
        return result
    
    def detect_signal_reversal(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect signal reversals (flip from long to short or vice versa).
        
        Args:
            df: DataFrame with signal column
            
        Returns:
            Boolean series indicating reversal points
        """
        signal_change = df['signal'] != df['signal'].shift(1)
        signal_flip = (
            ((df['signal'] == 1) & (df['signal'].shift(1) == -1)) |
            ((df['signal'] == -1) & (df['signal'].shift(1) == 1))
        )
        
        return signal_change & signal_flip
    
    def detect_signal_weakening(self, df: pd.DataFrame, 
                               threshold_pct: float = 0.3) -> pd.Series:
        """
        Detect when signal strength is weakening significantly.
        
        Args:
            df: DataFrame with signal_strength column
            threshold_pct: Percentage threshold for weakening detection
            
        Returns:
            Boolean series indicating weakening signals
        """
        strength_change = df['signal_strength'].pct_change()
        weakening = strength_change < -threshold_pct
        
        return weakening


def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Convenience function to generate signals from DataFrame.
    
    Args:
        df: DataFrame with all indicator columns
        params: Strategy parameters dictionary
        
    Returns:
        DataFrame with signals added
    """
    generator = SignalGenerator(params)
    result = generator.generate_signals(df)
    
    # Add reversal and weakening detection
    result['signal_reversal'] = generator.detect_signal_reversal(result)
    result['signal_weakening'] = generator.detect_signal_weakening(result)
    
    return result


def calculate_entry_price(df: pd.DataFrame, signal: int, 
                         slippage: float = 0.0005) -> float:
    """
    Calculate entry price with slippage.
    
    Args:
        df: DataFrame row with OHLC data
        signal: 1 for long, -1 for short
        slippage: Slippage percentage
        
    Returns:
        Entry price adjusted for slippage
    """
    close_price = df['close']
    
    if signal == 1:  # Long entry
        entry_price = close_price * (1 + slippage)
    elif signal == -1:  # Short entry
        entry_price = close_price * (1 - slippage)
    else:
        entry_price = close_price
    
    return entry_price


def validate_signal(df: pd.DataFrame, idx: int, params: Dict) -> bool:
    """
    Validate if a signal at given index meets all conditions.
    
    Performs additional validation checks beyond basic scoring:
    - Ensures indicator values are not NaN
    - Checks for minimum data requirements
    - Validates signal consistency
    
    Args:
        df: DataFrame with all columns
        idx: Index to validate
        params: Strategy parameters
        
    Returns:
        True if signal is valid, False otherwise
    """
    if idx < 0 or idx >= len(df):
        return False
    
    row = df.iloc[idx]
    
    # Check for NaN values in critical indicators
    critical_cols = [
        'adx', 'composite_score', 'adjusted_score', 'signal'
    ]
    
    for col in critical_cols:
        if col in row and pd.isna(row[col]):
            return False
    
    # Check if signal is valid (not 0)
    if row['signal'] == 0:
        return False
    
    # Check minimum lookback period has passed
    min_lookback = max(
        params.get('adx_length', 14),
        params.get('regime_hma_length', 21),
        params.get('linreg_length', 50)
    )
    
    if idx < min_lookback:
        return False
    
    return True
