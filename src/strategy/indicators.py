"""
QTAlgo Super26 Strategy - Technical Indicators Module

This module implements all 7 technical indicators used in the Super26 strategy:
1. ADX (Average Directional Index) - Trend strength filter
2. Regime Filter - Trend and volume analysis using HMA
3. Pivot Trend - Primary signal generator using pivot highs/lows
4. Trend Duration Forecast - HMA-based trend detection
5. ML Adaptive SuperTrend - Volatility-adaptive SuperTrend
6. Linear Regression Channel - Slope-based trend analysis
7. Pivot Levels - Support/resistance confirmation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


def hma(series: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average (HMA)
    
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    Args:
        series: Input price series
        length: HMA period length
        
    Returns:
        Hull Moving Average series
    """
    if length < 1:
        return series
    
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    
    wma_half = series.rolling(window=half_length).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    wma_full = series.rolling(window=length).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    
    raw_hma = 2 * wma_half - wma_full
    
    hma_result = raw_hma.rolling(window=sqrt_length).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    
    return hma_result


def calculate_adx(df: pd.DataFrame, adxlen: int = 14, dilen: int = 14) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index) and related indicators
    
    Args:
        df: DataFrame with OHLC data (must have 'high', 'low', 'close' columns)
        adxlen: ADX smoothing length
        dilen: DI smoothing length
        
    Returns:
        DataFrame with ADX, DI+, DI-, trend_strength columns
    """
    result = df.copy()
    
    # Calculate True Range and Directional Movement
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Smoothed TR and DM
    atr = tr.rolling(window=dilen).mean()
    plus_di_smooth = plus_dm.rolling(window=dilen).mean()
    minus_di_smooth = minus_dm.rolling(window=dilen).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_di_smooth / atr)
    minus_di = 100 * (minus_di_smooth / atr)
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=adxlen).mean()
    
    result['adx'] = adx
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    
    # Trend strength classification
    result['adx_trend_strength'] = np.where(adx >= 25, 'strong',
                                             np.where(adx >= 20, 'moderate', 'weak'))
    
    return result


def calculate_regime_filter(df: pd.DataFrame, length: int = 50, mult: float = 2.0) -> pd.DataFrame:
    """
    Regime Filter - Trend and volume analysis using HMA
    
    Args:
        df: DataFrame with OHLC data
        length: HMA length
        mult: Multiplier for regime bands
        
    Returns:
        DataFrame with regime filter signals
    """
    result = df.copy()
    
    # Calculate HMA of close price
    hma_close = hma(df['close'], length)
    
    # Calculate standard deviation
    std_dev = df['close'].rolling(window=length).std()
    
    # Upper and lower bands
    upper_band = hma_close + (mult * std_dev)
    lower_band = hma_close - (mult * std_dev)
    
    # Regime signal
    regime = np.where(df['close'] > upper_band, 1,  # Bullish regime
                      np.where(df['close'] < lower_band, -1,  # Bearish regime
                               0))  # Neutral regime
    
    result['regime_hma'] = hma_close
    result['regime_upper'] = upper_band
    result['regime_lower'] = lower_band
    result['regime_signal'] = regime
    
    return result


def calculate_pivot_trend(df: pd.DataFrame, pivot_length: int = 10, 
                         atr_length: int = 14, atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Pivot Trend - Primary signal generator using pivot highs/lows
    
    Args:
        df: DataFrame with OHLC data
        pivot_length: Length for pivot detection
        atr_length: ATR length for offset
        atr_mult: ATR multiplier
        
    Returns:
        DataFrame with pivot trend signals
    """
    result = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR for offset
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length).mean()
    
    # Pivot High detection
    pivot_high = high.rolling(window=pivot_length * 2 + 1, center=True).apply(
        lambda x: x[pivot_length] if len(x) == pivot_length * 2 + 1 and 
        x[pivot_length] == max(x) else np.nan,
        raw=True
    )
    
    # Pivot Low detection
    pivot_low = low.rolling(window=pivot_length * 2 + 1, center=True).apply(
        lambda x: x[pivot_length] if len(x) == pivot_length * 2 + 1 and 
        x[pivot_length] == min(x) else np.nan,
        raw=True
    )
    
    # Fill forward pivot values and apply ATR offset
    pivot_high_level = pivot_high.ffill() + (atr * atr_mult)
    pivot_low_level = pivot_low.ffill() - (atr * atr_mult)
    
    # Trend signal
    trend = np.where(close > pivot_high_level, 1,  # Bullish
                     np.where(close < pivot_low_level, -1,  # Bearish
                              0))  # Neutral
    
    result['pivot_high'] = pivot_high_level
    result['pivot_low'] = pivot_low_level
    result['pivot_trend'] = trend
    
    return result


def calculate_trend_duration(df: pd.DataFrame, hma_length: int = 20, 
                            trend_length: int = 10) -> pd.DataFrame:
    """
    Trend Duration Forecast - HMA-based trend detection
    
    Args:
        df: DataFrame with OHLC data
        hma_length: HMA length
        trend_length: Trend duration lookback
        
    Returns:
        DataFrame with trend duration signals
    """
    result = df.copy()
    
    # Calculate HMA
    hma_close = hma(df['close'], hma_length)
    
    # Trend direction
    trend_up = hma_close > hma_close.shift(1)
    trend_down = hma_close < hma_close.shift(1)
    
    # Count consecutive trend bars
    trend_duration = pd.Series(0, index=df.index)
    current_duration = 0
    current_direction = 0
    
    for i in range(1, len(df)):
        if trend_up.iloc[i]:
            if current_direction == 1:
                current_duration += 1
            else:
                current_duration = 1
                current_direction = 1
        elif trend_down.iloc[i]:
            if current_direction == -1:
                current_duration += 1
            else:
                current_duration = 1
                current_direction = -1
        else:
            current_duration = 0
            current_direction = 0
        
        trend_duration.iloc[i] = current_duration * current_direction
    
    # Trend strength based on duration
    avg_duration = abs(trend_duration).rolling(window=trend_length).mean()
    
    result['trend_hma'] = hma_close
    result['trend_duration'] = trend_duration
    result['trend_strength'] = avg_duration
    result['trend_signal'] = np.sign(trend_duration)
    
    return result


def calculate_ml_supertrend(df: pd.DataFrame, atr_period: int = 10, 
                           factor: float = 3.0, lookback: int = 100) -> pd.DataFrame:
    """
    ML Adaptive SuperTrend - Volatility-adaptive SuperTrend
    
    Args:
        df: DataFrame with OHLC data
        atr_period: ATR period
        factor: SuperTrend factor
        lookback: Lookback for volatility percentile
        
    Returns:
        DataFrame with ML SuperTrend signals
    """
    result = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    
    # Volatility percentile adaptation
    volatility_percentile = atr.rolling(window=lookback).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5,
        raw=False
    )
    
    # Adaptive factor
    adaptive_factor = factor * (0.5 + volatility_percentile)
    
    # Basic upper and lower bands
    hl_avg = (high + low) / 2
    basic_upper = hl_avg + (adaptive_factor * atr)
    basic_lower = hl_avg - (adaptive_factor * atr)
    
    # SuperTrend calculation
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0, index=df.index)
    
    for i in range(1, len(df)):
        # Upper band
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        # Lower band
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # Trend
        if supertrend.iloc[i-1] == 1 and close.iloc[i] <= final_lower.iloc[i]:
            supertrend.iloc[i] = -1
        elif supertrend.iloc[i-1] == -1 and close.iloc[i] >= final_upper.iloc[i]:
            supertrend.iloc[i] = 1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            if supertrend.iloc[i] == 0:
                supertrend.iloc[i] = 1 if close.iloc[i] > hl_avg.iloc[i] else -1
    
    result['ml_supertrend_upper'] = final_upper
    result['ml_supertrend_lower'] = final_lower
    result['ml_supertrend'] = supertrend
    result['ml_supertrend_adaptive_factor'] = adaptive_factor
    
    return result


def calculate_linreg_channel(df: pd.DataFrame, length: int = 50, 
                             deviation: float = 2.0) -> pd.DataFrame:
    """
    Linear Regression Channel - Slope-based trend analysis
    
    Args:
        df: DataFrame with OHLC data
        length: Linear regression length
        deviation: Standard deviation multiplier
        
    Returns:
        DataFrame with linear regression channel
    """
    result = df.copy()
    
    close = df['close']
    
    # Calculate linear regression
    def linreg(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (len(y) - 1)
    
    def linreg_slope(y):
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    linreg_value = close.rolling(window=length).apply(linreg, raw=True)
    linreg_slope_value = close.rolling(window=length).apply(linreg_slope, raw=True)
    
    # Standard deviation for channel
    std_dev = close.rolling(window=length).std()
    
    upper_channel = linreg_value + (deviation * std_dev)
    lower_channel = linreg_value - (deviation * std_dev)
    
    # Trend signal based on slope and position
    slope_signal = np.where(linreg_slope_value > 0, 1,
                           np.where(linreg_slope_value < 0, -1, 0))
    
    position_signal = np.where(close > upper_channel, 1,
                              np.where(close < lower_channel, -1, 0))
    
    result['linreg'] = linreg_value
    result['linreg_upper'] = upper_channel
    result['linreg_lower'] = lower_channel
    result['linreg_slope'] = linreg_slope_value
    result['linreg_signal'] = slope_signal
    
    return result


def calculate_pivot_levels(df: pd.DataFrame, pivot_type: str = 'traditional',
                          atr_proximity: float = 1.5) -> pd.DataFrame:
    """
    Pivot Levels - Support/resistance confirmation
    
    Args:
        df: DataFrame with OHLC data
        pivot_type: Type of pivot calculation ('traditional', 'fibonacci', 'camarilla')
        atr_proximity: ATR multiplier for proximity check
        
    Returns:
        DataFrame with pivot levels and signals
    """
    result = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate ATR for proximity
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Traditional Pivot Points (daily)
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    
    if pivot_type == 'traditional':
        r1 = 2 * pivot - low.shift(1)
        r2 = pivot + (high.shift(1) - low.shift(1))
        s1 = 2 * pivot - high.shift(1)
        s2 = pivot - (high.shift(1) - low.shift(1))
    elif pivot_type == 'fibonacci':
        r1 = pivot + 0.382 * (high.shift(1) - low.shift(1))
        r2 = pivot + 0.618 * (high.shift(1) - low.shift(1))
        s1 = pivot - 0.382 * (high.shift(1) - low.shift(1))
        s2 = pivot - 0.618 * (high.shift(1) - low.shift(1))
    else:  # camarilla
        r1 = close.shift(1) + 1.1 * (high.shift(1) - low.shift(1)) / 12
        r2 = close.shift(1) + 1.1 * (high.shift(1) - low.shift(1)) / 6
        s1 = close.shift(1) - 1.1 * (high.shift(1) - low.shift(1)) / 12
        s2 = close.shift(1) - 1.1 * (high.shift(1) - low.shift(1)) / 6
    
    # Check proximity to pivot levels
    proximity_threshold = atr * atr_proximity
    
    near_resistance = ((abs(close - r1) < proximity_threshold) | 
                       (abs(close - r2) < proximity_threshold))
    near_support = ((abs(close - s1) < proximity_threshold) | 
                    (abs(close - s2) < proximity_threshold))
    
    # Signal: 1 if near support (bullish), -1 if near resistance (bearish), 0 otherwise
    pivot_signal = np.where(near_support, 1,
                           np.where(near_resistance, -1, 0))
    
    result['pivot_point'] = pivot
    result['pivot_r1'] = r1
    result['pivot_r2'] = r2
    result['pivot_s1'] = s1
    result['pivot_s2'] = s2
    result['pivot_signal'] = pivot_signal
    
    return result


def calculate_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate all indicators for the Super26 strategy
    
    Args:
        df: DataFrame with OHLC data
        params: Dictionary of strategy parameters
        
    Returns:
        DataFrame with all indicators calculated
    """
    result = df.copy()
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in result.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate all indicators
    result = calculate_adx(result, 
                          adxlen=params.get('adxlen', 14),
                          dilen=params.get('dilen', 14))
    
    result = calculate_regime_filter(result,
                                     length=params.get('regime_length', 50),
                                     mult=params.get('regime_mult', 2.0))
    
    result = calculate_pivot_trend(result,
                                   pivot_length=params.get('pivot_length', 10),
                                   atr_length=params.get('pivot_atr_length', 14),
                                   atr_mult=params.get('pivot_atr_mult', 1.0))
    
    result = calculate_trend_duration(result,
                                     hma_length=params.get('trend_hma_length', 20),
                                     trend_length=params.get('trend_length', 10))
    
    result = calculate_ml_supertrend(result,
                                    atr_period=params.get('ml_atr_period', 10),
                                    factor=params.get('ml_factor', 3.0),
                                    lookback=params.get('ml_lookback', 100))
    
    result = calculate_linreg_channel(result,
                                     length=params.get('linreg_length', 50),
                                     deviation=params.get('linreg_deviation', 2.0))
    
    result = calculate_pivot_levels(result,
                                   pivot_type=params.get('pivot_type', 'traditional'),
                                   atr_proximity=params.get('pivot_atr_proximity', 1.5))
    
    return result
