"""
Walk-Forward Optimization Engine

Implements both anchored and rolling window walk-forward optimization
with multi-objective optimization support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

from .parameter_space import ParameterSpace, merge_parameters
from .metrics import calculate_all_metrics, PerformanceMetrics, metrics_to_dict

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    best_params: Optional[Dict] = None
    train_metrics: Optional[PerformanceMetrics] = None
    test_metrics: Optional[PerformanceMetrics] = None


@dataclass
class OptimizationResult:
    """Results from a single optimization run."""
    parameters: Dict
    train_metrics: PerformanceMetrics
    test_metrics: Optional[PerformanceMetrics] = None
    objective_value: float = 0.0


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine for trading strategies.
    
    Supports:
    - Rolling and anchored window approaches
    - Multi-objective optimization
    - Parameter stability tracking
    - Walk-forward efficiency calculation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize walk-forward optimizer.
        
        Args:
            config: Configuration dictionary with optimization settings
        """
        self.config = config
        
        # Walk-forward settings
        self.mode = config.get('mode', 'rolling')
        self.train_period_months = config.get('train_period_months', 12)
        self.test_period_months = config.get('test_period_months', 3)
        self.step_months = config.get('step_months', 3)
        
        # Optimization settings
        self.algorithm = config.get('algorithm', 'optuna')
        self.n_trials = config.get('n_trials', 100)
        self.n_jobs = config.get('n_jobs', -1)
        
        # Objectives
        self.objectives = config.get('objectives', {
            'sharpe_ratio': {'weight': 0.4, 'direction': 'maximize'},
            'max_drawdown': {'weight': 0.3, 'direction': 'minimize'},
            'win_rate': {'weight': 0.2, 'direction': 'maximize'},
            'profit_factor': {'weight': 0.1, 'direction': 'maximize'}
        })
        
        # Results storage
        self.wf_periods: List[WalkForwardPeriod] = []
        self.all_results: List[OptimizationResult] = []
    
    def split_data(self, df: pd.DataFrame) -> List[WalkForwardPeriod]:
        """
        Split data into walk-forward periods.
        
        Args:
            df: Full historical data
            
        Returns:
            List of WalkForwardPeriod objects
        """
        periods = []
        
        # Sort data by index
        df = df.sort_index()
        
        # Calculate period lengths
        train_days = self.train_period_months * 30
        test_days = self.test_period_months * 30
        step_days = self.step_months * 30
        
        # Start with first train period
        current_train_start = df.index[0]
        
        while True:
            # Calculate train period
            train_end_idx = current_train_start + pd.Timedelta(days=train_days)
            
            if train_end_idx >= df.index[-1]:
                break
            
            # Calculate test period
            test_start = train_end_idx
            test_end = test_start + pd.Timedelta(days=test_days)
            
            if test_end > df.index[-1]:
                test_end = df.index[-1]
            
            # Extract data
            train_data = df[(df.index >= current_train_start) & (df.index < train_end_idx)]
            test_data = df[(df.index >= test_start) & (df.index <= test_end)]
            
            if len(train_data) < 100 or len(test_data) < 10:
                logger.warning(f"Insufficient data for period starting {current_train_start}")
                break
            
            period = WalkForwardPeriod(
                train_start=current_train_start,
                train_end=train_end_idx,
                test_start=test_start,
                test_end=test_end,
                train_data=train_data,
                test_data=test_data
            )
            
            periods.append(period)
            
            # Move to next period
            if self.mode == 'rolling':
                # Rolling window: move start forward
                current_train_start = current_train_start + pd.Timedelta(days=step_days)
            else:
                # Anchored window: keep start at beginning
                current_train_start = df.index[0]
                # But update train end for next iteration
                train_end_idx = test_end
                if train_end_idx >= df.index[-1]:
                    break
            
            # For anchored, we iterate differently
            if self.mode == 'anchored':
                # Next test period starts where this one ended
                if test_end >= df.index[-1]:
                    break
        
        logger.info(f"Created {len(periods)} walk-forward periods ({self.mode} mode)")
        
        return periods
    
    def calculate_objective(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate multi-objective score.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Combined objective score
        """
        score = 0.0
        metrics_dict = metrics_to_dict(metrics)
        
        for metric_name, config in self.objectives.items():
            weight = config['weight']
            direction = config['direction']
            
            value = metrics_dict.get(metric_name, 0.0)
            
            # Handle infinities
            if np.isinf(value):
                value = 1e6 if value > 0 else -1e6
            
            # Apply direction
            if direction == 'minimize':
                value = -value
            
            score += weight * value
        
        return score
    
    def optimize_period(self, period: WalkForwardPeriod,
                       param_space: ParameterSpace,
                       strategy_func: Callable) -> WalkForwardPeriod:
        """
        Optimize parameters for a single period.
        
        Args:
            period: WalkForwardPeriod to optimize
            param_space: Parameter search space
            strategy_func: Function that takes (data, params) and returns trades_df
            
        Returns:
            Updated WalkForwardPeriod with best parameters and metrics
        """
        logger.info(f"Optimizing period: {period.train_start} to {period.train_end}")
        
        if self.algorithm == 'optuna':
            best_params, train_metrics = self._optimize_with_optuna(
                period.train_data,
                param_space,
                strategy_func
            )
        elif self.algorithm == 'random_search':
            best_params, train_metrics = self._optimize_with_random_search(
                period.train_data,
                param_space,
                strategy_func
            )
        else:
            raise ValueError(f"Unknown optimization algorithm: {self.algorithm}")
        
        # Test on out-of-sample data
        trades_df = strategy_func(period.test_data, best_params)
        test_metrics = calculate_all_metrics(trades_df)
        
        # Update period
        period.best_params = best_params
        period.train_metrics = train_metrics
        period.test_metrics = test_metrics
        
        logger.info(
            f"Period optimized - Train Sharpe: {train_metrics.sharpe_ratio:.2f}, "
            f"Test Sharpe: {test_metrics.sharpe_ratio:.2f}"
        )
        
        return period
    
    def _optimize_with_optuna(self, train_data: pd.DataFrame,
                             param_space: ParameterSpace,
                             strategy_func: Callable) -> Tuple[Dict, PerformanceMetrics]:
        """
        Optimize using Optuna.
        
        Args:
            train_data: Training data
            param_space: Parameter space
            strategy_func: Strategy function
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        import optuna
        
        def objective(trial):
            # Sample parameters
            params = param_space.create_optuna_space(trial)
            
            # Run strategy
            try:
                trades_df = strategy_func(train_data, params)
                metrics = calculate_all_metrics(trades_df)
                
                # Calculate objective
                score = self.calculate_objective(metrics)
                
                return score
            
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -1e6  # Return very bad score on failure
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False
        )
        
        # Get best parameters
        best_params = study.best_params
        
        # Calculate metrics with best parameters
        trades_df = strategy_func(train_data, best_params)
        best_metrics = calculate_all_metrics(trades_df)
        
        return best_params, best_metrics
    
    def _optimize_with_random_search(self, train_data: pd.DataFrame,
                                    param_space: ParameterSpace,
                                    strategy_func: Callable) -> Tuple[Dict, PerformanceMetrics]:
        """
        Optimize using random search.
        
        Args:
            train_data: Training data
            param_space: Parameter space
            strategy_func: Strategy function
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Generate random samples
        samples = param_space.sample_random(self.n_trials)
        
        for params in samples:
            try:
                # Run strategy
                trades_df = strategy_func(train_data, params)
                metrics = calculate_all_metrics(trades_df)
                
                # Calculate objective
                score = self.calculate_objective(metrics)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
            
            except Exception as e:
                logger.warning(f"Sample failed: {e}")
                continue
        
        if best_params is None:
            # Use default parameters if all failed
            best_params = param_space.get_default_parameters()
            trades_df = strategy_func(train_data, best_params)
            best_metrics = calculate_all_metrics(trades_df)
        
        return best_params, best_metrics
    
    def run_walk_forward(self, df: pd.DataFrame,
                        param_space: ParameterSpace,
                        strategy_func: Callable) -> List[WalkForwardPeriod]:
        """
        Run complete walk-forward optimization.
        
        Args:
            df: Full historical data
            param_space: Parameter search space
            strategy_func: Strategy function
            
        Returns:
            List of optimized WalkForwardPeriod objects
        """
        # Split data into periods
        periods = self.split_data(df)
        
        # Optimize each period
        optimized_periods = []
        for i, period in enumerate(periods):
            logger.info(f"Processing period {i+1}/{len(periods)}")
            optimized_period = self.optimize_period(period, param_space, strategy_func)
            optimized_periods.append(optimized_period)
        
        self.wf_periods = optimized_periods
        
        return optimized_periods
    
    def calculate_wfe(self) -> float:
        """
        Calculate Walk-Forward Efficiency (WFE).
        
        WFE = (Total OOS Return) / (Total IS Return)
        
        Returns:
            Walk-forward efficiency ratio
        """
        if not self.wf_periods:
            return 0.0
        
        total_is_return = sum(p.train_metrics.total_return for p in self.wf_periods 
                             if p.train_metrics)
        total_oos_return = sum(p.test_metrics.total_return for p in self.wf_periods 
                              if p.test_metrics)
        
        if total_is_return == 0:
            return 0.0
        
        wfe = total_oos_return / total_is_return
        
        return wfe
    
    def analyze_parameter_stability(self) -> pd.DataFrame:
        """
        Analyze parameter stability across periods.
        
        Returns:
            DataFrame with parameter statistics
        """
        if not self.wf_periods:
            return pd.DataFrame()
        
        # Collect all parameters
        all_params = []
        for period in self.wf_periods:
            if period.best_params:
                all_params.append(period.best_params)
        
        if not all_params:
            return pd.DataFrame()
        
        # Create DataFrame
        params_df = pd.DataFrame(all_params)
        
        # Calculate statistics
        stats = pd.DataFrame({
            'mean': params_df.mean(),
            'std': params_df.std(),
            'min': params_df.min(),
            'max': params_df.max(),
            'cv': params_df.std() / params_df.mean()  # Coefficient of variation
        })
        
        return stats
    
    def get_aggregate_results(self) -> Dict:
        """
        Get aggregated results across all periods.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.wf_periods:
            return {}
        
        # Aggregate IS metrics
        is_metrics = [p.train_metrics for p in self.wf_periods if p.train_metrics]
        # Aggregate OOS metrics
        oos_metrics = [p.test_metrics for p in self.wf_periods if p.test_metrics]
        
        def aggregate(metrics_list):
            if not metrics_list:
                return {}
            return {
                'avg_sharpe': np.mean([m.sharpe_ratio for m in metrics_list]),
                'avg_return': np.mean([m.total_return for m in metrics_list]),
                'avg_drawdown': np.mean([m.max_drawdown for m in metrics_list]),
                'avg_win_rate': np.mean([m.win_rate for m in metrics_list]),
                'total_trades': sum([m.total_trades for m in metrics_list])
            }
        
        results = {
            'in_sample': aggregate(is_metrics),
            'out_of_sample': aggregate(oos_metrics),
            'wfe': self.calculate_wfe(),
            'num_periods': len(self.wf_periods)
        }
        
        return results
    
    def save_results(self, output_path: str) -> None:
        """
        Save optimization results to file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'periods': self.wf_periods,
                'aggregate_results': self.get_aggregate_results()
            }, f)
        
        logger.info(f"Results saved to {output_path}")
    
    def load_results(self, input_path: str) -> None:
        """
        Load optimization results from file.
        
        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.config = data['config']
        self.wf_periods = data['periods']
        
        logger.info(f"Results loaded from {input_path}")
