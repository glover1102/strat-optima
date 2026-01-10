"""
Technical Indicators for QTAlgo Super26 Strategy

This module implements all 7 core indicators used in the strategy:
1. ADX (Trend Strength Filter)
2. Regime Filter (HMA-based)
3. Pivot Trend
4. Trend Duration Forecast
5. ML Adaptive SuperTrend
6. Linear Regression Channel
7. Pivot Levels
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from numba import jit


def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    """
    Calculate Hull Moving Average (HMA).
    
    HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    
    Args:
        series: Input price series
        length: HMA period length
        
    Returns:
        Hull Moving Average series
    """
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = series.rolling(window=half_length).apply(
        lambda x: np.sum(x * np.arange(1, half_length + 1)) / np.sum(np.arange(1, half_length + 1)),
        raw=True
    )
    
    wma_full = series.rolling(window=length).apply(
        lambda x: np.sum(x * np.arange(1, length + 1)) / np.sum(np.arange(1, length + 1)),
        raw=True
    )
    
    diff = 2 * wma_half - wma_full
    
    hma = diff.rolling(window=sqrt_length).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_length + 1)) / np.sum(np.arange(1, sqrt_length + 1)),
        raw=True
    )
    
    return hma


def calculate_adx(df: pd.DataFrame, length: int = 14, di_length: int = 14, 
                  smoothing: int = 14, threshold: int = 20) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index) with trend strength scoring.
    
    ADX measures trend strength and provides a scoring system based on
    directional movement and threshold comparison.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        length: ADX calculation length
        di_length: Directional Indicator length
        smoothing: Smoothing period
        threshold: ADX threshold for trend strength
        
    Returns:
        DataFrame with ADX, +DI, -DI, and score columns
    """
    result = df.copy()
    
    # Calculate True Range
    high_low = result['high'] - result['low']
    high_close = np.abs(result['high'] - result['close'].shift(1))
    low_close = np.abs(result['low'] - result['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = result['high'] - result['high'].shift(1)
    down_move = result['low'].shift(1) - result['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    atr = pd.Series(tr).rolling(window=length).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=di_length).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=di_length).mean()
    
    # Calculate DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=smoothing).mean()
    
    # Calculate score
    score = np.where(
        adx > threshold,
        np.where(plus_di > minus_di, 1, -1),
        0
    )
    
    result['adx'] = adx
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    result['adx_score'] = score
    
    return result


def calculate_regime_filter(df: pd.DataFrame, hma_length: int = 21, 
                           volume_length: int = 20, 
                           volume_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Calculate Regime Filter using HMA and volume analysis.
    
    Identifies market regime (trending up/down or ranging) based on HMA
    slope and volume conditions.
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
        hma_length: Hull Moving Average length
        volume_length: Volume MA length
        volume_multiplier: Volume threshold multiplier
        
    Returns:
        DataFrame with regime filter score
    """
    result = df.copy()
    
    # Calculate HMA
    hma = calculate_hma(result['close'], hma_length)
    result['hma'] = hma
    
    # Calculate HMA slope
    hma_slope = hma - hma.shift(1)
    
    # Volume analysis
    volume_ma = result['volume'].rolling(window=volume_length).mean()
    high_volume = result['volume'] > (volume_ma * volume_multiplier)
    
    # Regime scoring
    score = np.where(
        high_volume,
        np.where(hma_slope > 0, 1, np.where(hma_slope < 0, -1, 0)),
        0
    )
    
    result['regime_score'] = score
    result['hma_slope'] = hma_slope
    
    return result


def calculate_pivot_trend(df: pd.DataFrame, left_bars: int = 4, 
                         right_bars: int = 2, atr_length: int = 14,
                         atr_multiplier: float = 0.5) -> pd.DataFrame:
    """
    Calculate Pivot Trend indicator using pivot highs/lows with ATR offset.
    
    Primary signal generator that identifies trend direction based on
    pivot points with ATR-based dynamic offset.
    
    Args:
        df: DataFrame with OHLC data
        left_bars: Left bars for pivot detection
        right_bars: Right bars for pivot detection
        atr_length: ATR calculation length
        atr_multiplier: ATR multiplier for offset
        
    Returns:
        DataFrame with pivot trend signals
    """
    result = df.copy()
    
    # Calculate ATR
    high_low = result['high'] - result['low']
    high_close = np.abs(result['high'] - result['close'].shift(1))
    low_close = np.abs(result['low'] - result['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    
    # Find pivot highs and lows
    pivot_high = result['high'].rolling(window=left_bars + right_bars + 1, center=True).apply(
        lambda x: x[left_bars] if x[left_bars] == max(x) else np.nan,
        raw=True
    )
    
    pivot_low = result['low'].rolling(window=left_bars + right_bars + 1, center=True).apply(
        lambda x: x[left_bars] if x[left_bars] == min(x) else np.nan,
        raw=True
    )
    
    # Apply ATR offset
    offset = atr * atr_multiplier
    pivot_high_adj = pivot_high + offset
    pivot_low_adj = pivot_low - offset
    
    # Forward fill pivots
    pivot_high_adj = pivot_high_adj.fillna(method='ffill')
    pivot_low_adj = pivot_low_adj.fillna(method='ffill')
    
    # Generate signals
    score = np.where(
        result['close'] > pivot_high_adj, 1,
        np.where(result['close'] < pivot_low_adj, -1, 0)
    )
    
    result['pivot_trend_score'] = score
    result['pivot_high'] = pivot_high_adj
    result['pivot_low'] = pivot_low_adj
    
    return result


def calculate_trend_duration(df: pd.DataFrame, hma_length: int = 34,
                            smoothing: int = 8) -> pd.DataFrame:
    """
    Calculate Trend Duration Forecast using HMA-based trend detection.
    
    Estimates trend persistence and strength using Hull Moving Average
    with smoothing.
    
    Args:
        df: DataFrame with 'close' column
        hma_length: HMA length for trend detection
        smoothing: Smoothing period
        
    Returns:
        DataFrame with trend duration score
    """
    result = df.copy()
    
    # Calculate HMA
    hma = calculate_hma(result['close'], hma_length)
    
    # Smooth HMA
    hma_smooth = hma.rolling(window=smoothing).mean()
    
    # Calculate trend
    trend = np.where(
        result['close'] > hma_smooth, 1,
        np.where(result['close'] < hma_smooth, -1, 0)
    )
    
    # Calculate trend duration (consecutive bars in same direction)
    trend_series = pd.Series(trend)
    duration = trend_series.groupby(
        (trend_series != trend_series.shift()).cumsum()
    ).cumcount() + 1
    
    # Score based on trend and duration
    score = np.where(
        duration > smoothing,
        trend,
        0
    )
    
    result['trend_duration_score'] = score
    result['trend_hma'] = hma_smooth
    result['trend_duration'] = duration
    
    return result


def calculate_ml_supertrend(df: pd.DataFrame, length: int = 10,
                           multiplier: float = 3.0, atr_length: int = 14,
                           volatility_window: int = 100) -> pd.DataFrame:
    """
    Calculate ML Adaptive SuperTrend with volatility percentile adjustment.
    
    Volatility-adaptive SuperTrend that adjusts multiplier based on
    current volatility percentile.
    
    Args:
        df: DataFrame with OHLC data
        length: SuperTrend length
        multiplier: Base multiplier
        atr_length: ATR calculation length
        volatility_window: Window for volatility percentile
        
    Returns:
        DataFrame with ML SuperTrend signals
    """
    result = df.copy()
    
    # Calculate ATR
    high_low = result['high'] - result['low']
    high_close = np.abs(result['high'] - result['close'].shift(1))
    low_close = np.abs(result['low'] - result['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    
    # Calculate volatility percentile
    volatility = result['close'].pct_change().rolling(window=length).std()
    vol_percentile = volatility.rolling(window=volatility_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )
    
    # Adaptive multiplier
    adaptive_mult = multiplier * (1 + vol_percentile)
    
    # Calculate HL average
    hl_avg = (result['high'] + result['low']) / 2
    
    # Calculate bands
    upper_band = hl_avg + (adaptive_mult * atr)
    lower_band = hl_avg - (adaptive_mult * atr)
    
    # Initialize SuperTrend
    supertrend = pd.Series(index=result.index, dtype=float)
    direction = pd.Series(index=result.index, dtype=int)
    
    # First value
    supertrend.iloc[0] = lower_band.iloc[0]
    direction.iloc[0] = 1
    
    # Calculate SuperTrend
    for i in range(1, len(result)):
        if result['close'].iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif result['close'].iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
            elif direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
    
    result['ml_supertrend'] = supertrend
    result['ml_supertrend_score'] = direction
    
    return result


def calculate_linear_regression_channel(df: pd.DataFrame, length: int = 50,
                                        deviation: float = 2.0,
                                        slope_threshold: float = 0.0001) -> pd.DataFrame:
    """
    Calculate Linear Regression Channel with slope-based trend analysis.
    
    Uses linear regression to identify trend direction and strength
    based on channel slope and price position.
    
    Args:
        df: DataFrame with 'close' column
        length: Linear regression length
        deviation: Standard deviation multiplier for bands
        slope_threshold: Minimum slope for trend identification
        
    Returns:
        DataFrame with linear regression channel signals
    """
    result = df.copy()
    
    def calc_linreg(y):
        """Calculate linear regression for a window."""
        if len(y) < 2:
            return np.nan, np.nan, np.nan
        x = np.arange(len(y))
        # Linear regression
        coef = np.polyfit(x, y, 1)
        line = np.polyval(coef, x)
        slope = coef[0]
        last_value = line[-1]
        # Standard deviation
        std = np.std(y - line)
        return last_value, slope, std
    
    linreg_vals = result['close'].rolling(window=length).apply(
        lambda x: calc_linreg(x)[0],
        raw=True
    )
    
    slopes = result['close'].rolling(window=length).apply(
        lambda x: calc_linreg(x)[1],
        raw=True
    )
    
    stds = result['close'].rolling(window=length).apply(
        lambda x: calc_linreg(x)[2],
        raw=True
    )
    
    # Calculate bands
    upper_band = linreg_vals + (stds * deviation)
    lower_band = linreg_vals - (stds * deviation)
    
    # Score based on slope and position
    score = np.where(
        slopes > slope_threshold, 1,
        np.where(slopes < -slope_threshold, -1, 0)
    )
    
    result['linreg_mid'] = linreg_vals
    result['linreg_upper'] = upper_band
    result['linreg_lower'] = lower_band
    result['linreg_slope'] = slopes
    result['linreg_score'] = score
    
    return result


def calculate_pivot_levels(df: pd.DataFrame, atr_multiplier: float = 1.5,
                          lookback: int = 20) -> pd.DataFrame:
    """
    Calculate Pivot Levels for support/resistance confirmation.
    
    Identifies key pivot levels and checks price proximity using ATR-based
    threshold for confirmation signals.
    
    Args:
        df: DataFrame with OHLC data
        atr_multiplier: ATR multiplier for proximity threshold
        lookback: Lookback period for pivot identification
        
    Returns:
        DataFrame with pivot level scores
    """
    result = df.copy()
    
    # Calculate ATR
    high_low = result['high'] - result['low']
    high_close = np.abs(result['high'] - result['close'].shift(1))
    low_close = np.abs(result['low'] - result['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Calculate pivot points (standard pivot calculation)
    pivot = (result['high'] + result['low'] + result['close']) / 3
    
    # Rolling high/low for support/resistance
    resistance = result['high'].rolling(window=lookback).max()
    support = result['low'].rolling(window=lookback).min()
    
    # Proximity threshold
    threshold = atr * atr_multiplier
    
    # Score based on proximity to levels
    near_resistance = np.abs(result['close'] - resistance) < threshold
    near_support = np.abs(result['close'] - support) < threshold
    
    score = np.where(
        near_resistance, -1,
        np.where(near_support, 1, 0)
    )
    
    result['pivot'] = pivot
    result['resistance'] = resistance
    result['support'] = support
    result['pivot_levels_score'] = score
    
    return result


def calculate_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate all indicators for the QTAlgo Super26 strategy.
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary of strategy parameters
        
    Returns:
        DataFrame with all indicator columns
    """
    result = df.copy()
    
    # 1. ADX
    result = calculate_adx(
        result,
        length=params.get('adx_length', 14),
        di_length=params.get('di_length', 14),
        smoothing=params.get('adx_smoothing', 14),
        threshold=params.get('adx_threshold', 20)
    )
    
    # 2. Regime Filter
    result = calculate_regime_filter(
        result,
        hma_length=params.get('regime_hma_length', 21),
        volume_length=params.get('regime_volume_length', 20),
        volume_multiplier=params.get('regime_volume_multiplier', 1.5)
    )
    
    # 3. Pivot Trend
    result = calculate_pivot_trend(
        result,
        left_bars=params.get('pivot_left_bars', 4),
        right_bars=params.get('pivot_right_bars', 2),
        atr_length=params.get('pivot_atr_length', 14),
        atr_multiplier=params.get('pivot_atr_multiplier', 0.5)
    )
    
    # 4. Trend Duration
    result = calculate_trend_duration(
        result,
        hma_length=params.get('trend_hma_length', 34),
        smoothing=params.get('trend_smoothing', 8)
    )
    
    # 5. ML Adaptive SuperTrend
    result = calculate_ml_supertrend(
        result,
        length=params.get('ml_length', 10),
        multiplier=params.get('ml_multiplier', 3.0),
        atr_length=params.get('ml_atr_length', 14),
        volatility_window=params.get('ml_volatility_window', 100)
    )
    
    # 6. Linear Regression Channel
    result = calculate_linear_regression_channel(
        result,
        length=params.get('linreg_length', 50),
        deviation=params.get('linreg_deviation', 2.0),
        slope_threshold=params.get('linreg_slope_threshold', 0.0001)
    )
    
    # 7. Pivot Levels
    result = calculate_pivot_levels(
        result,
        atr_multiplier=params.get('pivot_levels_atr_multiplier', 1.5),
        lookback=params.get('pivot_levels_lookback', 20)
    )
    
    return result
