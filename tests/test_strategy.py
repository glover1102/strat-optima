"""
Test Suite for QTAlgo Super26 Strategy

Basic tests for indicators, signals, and optimization components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategy.indicators import (
    calculate_adx,
    calculate_regime_filter,
    calculate_pivot_trend,
    hma
)
from src.strategy.signals import (
    generate_all_signals,
    calculate_indicator_scores
)
from src.strategy.exits import ExitManager, Position
from src.optimization.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_all_metrics
)
from src.optimization.parameter_space import ParameterSpace
from src.data.loader import DataLoader


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    n = len(dates)
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(n) * 0.02)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.3
    volume = np.random.randint(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def default_params():
    """Default strategy parameters"""
    return {
        'adxlen': 14,
        'dilen': 14,
        'adxThreshold': 20,
        'regime_length': 50,
        'regime_mult': 2.0,
        'pivot_length': 10,
        'pivot_atr_length': 14,
        'pivot_atr_mult': 1.0,
        'trend_hma_length': 20,
        'trend_length': 10,
        'ml_atr_period': 10,
        'ml_factor': 3.0,
        'ml_lookback': 100,
        'linreg_length': 50,
        'linreg_deviation': 2.0,
        'pivot_type': 'traditional',
        'pivot_atr_proximity': 1.5,
        'w_adx': 1.0,
        'w_regime': 1.0,
        'w_pivotTrend': 1.5,
        'w_trendDuration': 0.8,
        'w_mlSupertrend': 1.2,
        'w_linregChannel': 0.9,
        'w_pivotLevels': 0.7,
        'strongTrendMinScore': 1.5,
        'weakTrendMinScore': 3.0,
        'stopLossPercent': 2.0,
        'takeProfitPercent': 4.0,
        'partialExitPercent': 1.0,
        'trailingStopPercent': 0.8
    }


class TestIndicators:
    """Test indicator calculations"""
    
    def test_hma(self, sample_ohlcv_data):
        """Test Hull Moving Average"""
        hma_result = hma(sample_ohlcv_data['close'], 20)
        
        assert len(hma_result) == len(sample_ohlcv_data)
        assert not hma_result.iloc[-100:].isna().all()
    
    def test_adx(self, sample_ohlcv_data):
        """Test ADX calculation"""
        result = calculate_adx(sample_ohlcv_data.copy())
        
        assert 'adx' in result.columns
        assert 'plus_di' in result.columns
        assert 'minus_di' in result.columns
        assert not result['adx'].iloc[-100:].isna().all()
    
    def test_regime_filter(self, sample_ohlcv_data):
        """Test regime filter"""
        result = calculate_regime_filter(sample_ohlcv_data.copy())
        
        assert 'regime_signal' in result.columns
        assert result['regime_signal'].isin([-1, 0, 1]).all()
    
    def test_pivot_trend(self, sample_ohlcv_data):
        """Test pivot trend indicator"""
        result = calculate_pivot_trend(sample_ohlcv_data.copy())
        
        assert 'pivot_trend' in result.columns
        assert result['pivot_trend'].isin([-1, 0, 1]).all()


class TestSignals:
    """Test signal generation"""
    
    def test_signal_generation(self, sample_ohlcv_data, default_params):
        """Test complete signal generation"""
        from src.strategy.indicators import calculate_all_indicators
        
        df_with_indicators = calculate_all_indicators(sample_ohlcv_data.copy(), default_params)
        df_with_signals = generate_all_signals(df_with_indicators, default_params)
        
        assert 'entry_signal' in df_with_signals.columns
        assert 'signal_strength' in df_with_signals.columns
        assert 'total_score' in df_with_signals.columns
        assert df_with_signals['entry_signal'].isin([-1, 0, 1]).all()
    
    def test_indicator_scores(self, sample_ohlcv_data, default_params):
        """Test indicator scoring"""
        from src.strategy.indicators import calculate_all_indicators
        
        df_with_indicators = calculate_all_indicators(sample_ohlcv_data.copy(), default_params)
        
        weights = {
            'w_adx': 1.0,
            'w_regime': 1.0,
            'w_pivotTrend': 1.5,
            'w_trendDuration': 0.8,
            'w_mlSupertrend': 1.2,
            'w_linregChannel': 0.9,
            'w_pivotLevels': 0.7
        }
        
        df_with_scores = calculate_indicator_scores(df_with_indicators, weights)
        
        score_columns = ['score_adx', 'score_regime', 'score_pivot_trend']
        for col in score_columns:
            assert col in df_with_scores.columns


class TestExits:
    """Test exit management"""
    
    def test_exit_manager(self, default_params):
        """Test exit manager initialization"""
        exit_manager = ExitManager(default_params)
        
        assert exit_manager.stop_loss_percent == 0.02
        assert exit_manager.take_profit_percent == 0.04
    
    def test_calculate_exit_levels_long(self, default_params):
        """Test exit level calculation for long position"""
        exit_manager = ExitManager(default_params)
        
        position = Position(
            entry_price=100.0,
            entry_bar=0,
            direction=1
        )
        
        position = exit_manager.calculate_exit_levels(position)
        
        assert position.initial_stop == 98.0  # 2% below
        assert position.partial_exit_target == 101.0  # 1% above
        assert position.final_exit_target == 104.0  # 4% above
    
    def test_calculate_exit_levels_short(self, default_params):
        """Test exit level calculation for short position"""
        exit_manager = ExitManager(default_params)
        
        position = Position(
            entry_price=100.0,
            entry_bar=0,
            direction=-1
        )
        
        position = exit_manager.calculate_exit_levels(position)
        
        assert position.initial_stop == 102.0  # 2% above
        assert position.partial_exit_target == 99.0  # 1% below
        assert position.final_exit_target == 96.0  # 4% below


class TestMetrics:
    """Test performance metrics"""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_max_drawdown(self):
        """Test max drawdown calculation"""
        equity = pd.Series([100, 110, 105, 120, 100, 115, 125])
        max_dd, duration = calculate_max_drawdown(equity)
        
        assert max_dd > 0
        assert duration >= 0
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation"""
        # Create sample equity curve
        equity = pd.Series(100 + np.cumsum(np.random.randn(252) * 0.5))
        equity.index = pd.date_range(start='2020-01-01', periods=252, freq='D')
        
        # Create sample trades
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, 200, -30],
            'entry_date': pd.date_range(start='2020-01-01', periods=5, freq='M'),
            'exit_date': pd.date_range(start='2020-01-15', periods=5, freq='M')
        })
        
        metrics = calculate_all_metrics(equity, trades)
        
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        assert metrics.total_trades == 5


class TestParameterSpace:
    """Test parameter space management"""
    
    def test_parameter_space_creation(self):
        """Test parameter space initialization"""
        config = {
            'param1': [1.0, 5.0, 0.5],
            'param2': [10, 20, 1],
            'param3': [0.1, 1.0]
        }
        
        param_space = ParameterSpace(config)
        
        assert len(param_space.parameters) == 3
        assert 'param1' in param_space.parameters
    
    def test_sample_random(self):
        """Test random parameter sampling"""
        config = {
            'param1': [1.0, 5.0],
            'param2': [10, 20]
        }
        
        param_space = ParameterSpace(config)
        samples = param_space.sample_random(n_samples=10)
        
        assert len(samples) == 10
        for sample in samples:
            assert 1.0 <= sample['param1'] <= 5.0
            assert 10 <= sample['param2'] <= 20
    
    def test_validate_parameters(self):
        """Test parameter validation"""
        config = {
            'param1': [1.0, 5.0],
            'param2': [10, 20]
        }
        
        param_space = ParameterSpace(config)
        
        # Valid parameters
        assert param_space.validate_parameters({'param1': 3.0, 'param2': 15})
        
        # Invalid parameters
        assert not param_space.validate_parameters({'param1': 10.0, 'param2': 15})


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_data_validation(self, sample_ohlcv_data):
        """Test data validation"""
        loader = DataLoader(source='csv')
        
        assert loader.validate_data(sample_ohlcv_data)
    
    def test_handle_missing_data(self, sample_ohlcv_data):
        """Test missing data handling"""
        loader = DataLoader(source='csv')
        
        # Add some missing values
        df = sample_ohlcv_data.copy()
        df.iloc[10:15, 0] = np.nan
        
        df_filled = loader.handle_missing_data(df, method='ffill')
        
        assert not df_filled.isna().any().any()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
