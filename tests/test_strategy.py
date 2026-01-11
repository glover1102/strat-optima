"""
Unit Tests for QTAlgo Super26 Strategy Components

Tests core functionality of indicators, signals, and exits.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.strategy.indicators import (
    calculate_adx, calculate_regime_filter, calculate_pivot_trend,
    calculate_hma, calculate_all_indicators
)
from src.strategy.signals import SignalGenerator, generate_signals
from src.strategy.exits import ExitManager, Position, ExitReason
from src.data.loader import DataLoader
from src.optimization.metrics import (
    calculate_all_metrics, calculate_sharpe_ratio, calculate_win_rate
)
from src.optimization.parameter_space import ParameterSpace


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    
    # Generate synthetic price data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    high = close + np.random.rand(1000) * 2
    low = close - np.random.rand(1000) * 2
    open_price = close + np.random.randn(1000) * 0.5
    volume = np.random.randint(1000, 10000, 1000)
    
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
    """Default strategy parameters."""
    return {
        'strongTrendMinScore': 1.5,
        'weakTrendMinScore': 3.0,
        'stopLossPercent': 2.0,
        'takeProfitPercent': 4.0,
        'partialExitPercent': 1.0,
        'trailingStopPercent': 0.8,
        'adx_threshold': 20,
        'adx_length': 14,
        'di_length': 14,
        'adx_smoothing': 14,
        'w_adx': 1.0,
        'w_regime': 1.0,
        'w_pivotTrend': 1.0,
        'w_trendDuration': 1.0,
        'w_mlSupertrend': 1.0,
        'w_linReg': 1.0,
        'w_pivotLevels': 1.0,
    }


class TestIndicators:
    """Test indicator calculations."""
    
    def test_hma_calculation(self, sample_data):
        """Test Hull Moving Average calculation."""
        hma = calculate_hma(sample_data['close'], 21)
        
        assert len(hma) == len(sample_data)
        assert not hma.isna().all()
        assert hma.iloc[-1] != 0
    
    def test_adx_calculation(self, sample_data):
        """Test ADX calculation."""
        result = calculate_adx(sample_data)
        
        assert 'adx' in result.columns
        assert 'plus_di' in result.columns
        assert 'minus_di' in result.columns
        assert 'adx_score' in result.columns
        
        # Check values are reasonable
        assert result['adx'].max() <= 100
        assert result['adx'].min() >= 0
    
    def test_all_indicators(self, sample_data, default_params):
        """Test calculation of all indicators."""
        result = calculate_all_indicators(sample_data, default_params)
        
        required_cols = [
            'adx_score', 'regime_score', 'pivot_trend_score',
            'trend_duration_score', 'ml_supertrend_score',
            'linreg_score', 'pivot_levels_score'
        ]
        
        for col in required_cols:
            assert col in result.columns


class TestSignals:
    """Test signal generation."""
    
    def test_signal_generator_init(self, default_params):
        """Test signal generator initialization."""
        generator = SignalGenerator(default_params)
        
        assert generator.strong_trend_min == 1.5
        assert generator.weak_trend_min == 3.0
        assert generator.weights['adx'] == 1.0
    
    def test_generate_signals(self, sample_data, default_params):
        """Test signal generation."""
        # Add indicators first
        df_with_indicators = calculate_all_indicators(sample_data, default_params)
        
        # Generate signals
        result = generate_signals(df_with_indicators, default_params)
        
        assert 'signal' in result.columns
        assert 'composite_score' in result.columns
        assert 'signal_strength' in result.columns
        
        # Signals should be -1, 0, or 1
        assert result['signal'].isin([-1, 0, 1]).all()


class TestExits:
    """Test exit management."""
    
    def test_exit_manager_init(self, default_params):
        """Test exit manager initialization."""
        manager = ExitManager(default_params)
        
        assert manager.stop_loss_pct == 0.02
        assert manager.take_profit_pct == 0.04
    
    def test_open_long_position(self, default_params):
        """Test opening a long position."""
        manager = ExitManager(default_params)
        
        position = manager.open_position(
            entry_price=100.0,
            entry_time=pd.Timestamp('2023-01-01'),
            direction=1,
            size=1.0
        )
        
        assert position.direction == 1
        assert position.entry_price == 100.0
        assert position.initial_stop < 100.0
        assert position.partial_target > 100.0
        assert position.final_target > position.partial_target
    
    def test_check_stop_loss(self, default_params):
        """Test stop loss exit."""
        manager = ExitManager(default_params)
        
        position = manager.open_position(
            entry_price=100.0,
            entry_time=pd.Timestamp('2023-01-01'),
            direction=1,
            size=1.0
        )
        
        # Price hits stop loss
        should_exit, reason, exit_price = manager.check_exit(
            position, 
            current_price=position.initial_stop - 0.1
        )
        
        assert should_exit
        assert reason == ExitReason.STOP_LOSS


class TestDataLoader:
    """Test data loading."""
    
    def test_data_validation(self, sample_data):
        """Test data validation."""
        loader = DataLoader()
        
        is_valid, issues = loader.validate_data(sample_data)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_data_cleaning(self, sample_data):
        """Test data cleaning."""
        # Add some NaN values
        dirty_data = sample_data.copy()
        dirty_data.loc[dirty_data.index[10], 'close'] = np.nan
        
        loader = DataLoader()
        cleaned = loader.clean_data(dirty_data)
        
        assert not cleaned['close'].isna().any()


class TestMetrics:
    """Test performance metrics."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_win_rate(self):
        """Test win rate calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150]
        })
        
        win_rate = calculate_win_rate(trades)
        
        assert win_rate == 0.6  # 3 out of 5 wins
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=10, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=10, freq='D'),
            'entry_price': [100] * 10,
            'exit_price': [102, 98, 105, 99, 103, 101, 104, 97, 106, 102],
            'direction': [1] * 10,
            'size': [1.0] * 10,
            'pnl': [200, -200, 500, -100, 300, 100, 400, -300, 600, 200],
            'pnl_pct': [2, -2, 5, -1, 3, 1, 4, -3, 6, 2],
            'duration': [24] * 10,
            'exit_reason': ['take_profit'] * 10
        })
        
        metrics = calculate_all_metrics(trades)
        
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 7
        assert metrics.losing_trades == 3
        assert metrics.total_return > 0


class TestParameterSpace:
    """Test parameter space management."""
    
    def test_add_parameter(self):
        """Test adding a parameter."""
        space = ParameterSpace()
        space.add_parameter('test_param', 1.0, 10.0, step=0.5)
        
        assert 'test_param' in space.parameters
        assert space.parameters['test_param'].min_value == 1.0
        assert space.parameters['test_param'].max_value == 10.0
    
    def test_sample_random(self):
        """Test random sampling."""
        space = ParameterSpace()
        space.add_parameter('param1', 1.0, 10.0)
        space.add_parameter('param2', 0.5, 2.0)
        
        samples = space.sample_random(n_samples=5)
        
        assert len(samples) == 5
        for sample in samples:
            assert 'param1' in sample
            assert 'param2' in sample
            assert 1.0 <= sample['param1'] <= 10.0
            assert 0.5 <= sample['param2'] <= 2.0
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        space = ParameterSpace()
        space.add_parameter('param1', 1.0, 10.0)
        
        # Valid parameters
        is_valid, issues = space.validate_parameters({'param1': 5.0})
        assert is_valid
        assert len(issues) == 0
        
        # Invalid parameters (out of bounds)
        is_valid, issues = space.validate_parameters({'param1': 15.0})
        assert not is_valid
        assert len(issues) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
