"""
Walk-Forward Optimization Engine

Implements walk-forward analysis with both anchored and rolling window approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

from .metrics import calculate_all_metrics, compare_is_oos_metrics, metrics_to_dict, PerformanceMetrics
from .parameter_space import ParameterSpace, calculate_parameter_stability

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, float] = field(default_factory=dict)
    is_metrics: Optional[PerformanceMetrics] = None
    oos_metrics: Optional[PerformanceMetrics] = None
    

@dataclass
class WalkForwardResults:
    """Container for walk-forward optimization results"""
    windows: List[WalkForwardWindow]
    combined_oos_metrics: PerformanceMetrics
    parameter_stability: Dict[str, float]
    average_wfe: float
    total_windows: int
    optimization_time: float
    

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine
    
    Supports both anchored and rolling window approaches for robust
    out-of-sample validation.
    """
    
    def __init__(self, 
                 window_type: str = 'rolling',
                 train_period_months: int = 12,
                 test_period_months: int = 3,
                 step_size_months: int = 3,
                 min_data_points: int = 252):
        """
        Initialize walk-forward optimizer
        
        Args:
            window_type: 'rolling' or 'anchored'
            train_period_months: Training period in months
            test_period_months: Testing period in months
            step_size_months: Step size for rolling window
            min_data_points: Minimum data points required
        """
        self.window_type = window_type
        self.train_period_months = train_period_months
        self.test_period_months = test_period_months
        self.step_size_months = step_size_months
        self.min_data_points = min_data_points
    
    def generate_windows(self, df: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows
        
        Args:
            df: DataFrame with time-indexed data
            
        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        window_id = 0
        
        # Get date range
        start_date = df.index[0]
        end_date = df.index[-1]
        
        # Initial training period
        train_start = start_date
        train_end = train_start + pd.DateOffset(months=self.train_period_months)
        
        while True:
            # Test period
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_period_months)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Verify we have minimum data points
            train_data = df.loc[train_start:train_end]
            test_data = df.loc[test_start:test_end]
            
            if len(train_data) >= self.min_data_points and len(test_data) > 0:
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                )
                windows.append(window)
                window_id += 1
            
            # Move to next window
            if self.window_type == 'rolling':
                # Rolling: move both start and end
                train_start = train_start + pd.DateOffset(months=self.step_size_months)
                train_end = train_start + pd.DateOffset(months=self.train_period_months)
            else:
                # Anchored: keep start, extend end
                train_end = test_end
        
        logger.info(f"Generated {len(windows)} walk-forward windows ({self.window_type})")
        return windows
    
    def optimize_window(self,
                       train_data: pd.DataFrame,
                       param_space: ParameterSpace,
                       objective_func: Callable,
                       n_trials: int = 100,
                       algorithm: str = 'optuna') -> Dict[str, float]:
        """
        Optimize parameters for a single window
        
        Args:
            train_data: Training data
            param_space: Parameter space definition
            objective_func: Objective function to maximize
            n_trials: Number of optimization trials
            algorithm: Optimization algorithm
            
        Returns:
            Dictionary of optimal parameters
        """
        if algorithm == 'optuna':
            return self._optimize_with_optuna(train_data, param_space, objective_func, n_trials)
        elif algorithm == 'grid_search':
            return self._optimize_with_grid_search(train_data, param_space, objective_func)
        elif algorithm == 'random_search':
            return self._optimize_with_random_search(train_data, param_space, objective_func, n_trials)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _optimize_with_optuna(self,
                             train_data: pd.DataFrame,
                             param_space: ParameterSpace,
                             objective_func: Callable,
                             n_trials: int) -> Dict[str, float]:
        """Optimize using Optuna"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        
        def objective(trial):
            # Suggest parameters
            params = {}
            for name, param_def in param_space.parameters.items():
                if param_def.param_type == 'integer':
                    params[name] = trial.suggest_int(
                        name, 
                        int(param_def.min_value), 
                        int(param_def.max_value)
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        param_def.min_value,
                        param_def.max_value
                    )
            
            # Evaluate objective
            try:
                score = objective_func(train_data, params)
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -np.inf
        
        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _optimize_with_grid_search(self,
                                   train_data: pd.DataFrame,
                                   param_space: ParameterSpace,
                                   objective_func: Callable) -> Dict[str, float]:
        """Optimize using grid search"""
        grid = param_space.get_grid_points(n_points=5)
        
        best_score = -np.inf
        best_params = None
        
        for params in grid:
            try:
                score = objective_func(train_data, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.warning(f"Grid point failed: {e}")
        
        return best_params
    
    def _optimize_with_random_search(self,
                                    train_data: pd.DataFrame,
                                    param_space: ParameterSpace,
                                    objective_func: Callable,
                                    n_trials: int) -> Dict[str, float]:
        """Optimize using random search"""
        samples = param_space.sample_random(n_trials)
        
        best_score = -np.inf
        best_params = None
        
        for params in samples:
            try:
                score = objective_func(train_data, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.warning(f"Sample failed: {e}")
        
        return best_params
    
    def run_walk_forward(self,
                        df: pd.DataFrame,
                        param_space: ParameterSpace,
                        strategy_func: Callable,
                        objective_func: Callable,
                        n_trials: int = 100,
                        algorithm: str = 'optuna') -> WalkForwardResults:
        """
        Run complete walk-forward optimization
        
        Args:
            df: Full dataset
            param_space: Parameter space definition
            strategy_func: Function to run strategy and return equity curve
            objective_func: Objective function for optimization
            n_trials: Number of trials per window
            algorithm: Optimization algorithm
            
        Returns:
            WalkForwardResults object
        """
        start_time = datetime.now()
        
        # Generate windows
        windows = self.generate_windows(df)
        
        if len(windows) == 0:
            raise ValueError("No valid windows generated. Check data length and parameters.")
        
        # Optimize each window
        logger.info(f"Starting walk-forward optimization with {len(windows)} windows...")
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Get training and testing data
            train_data = df.loc[window.train_start:window.train_end]
            test_data = df.loc[window.test_start:window.test_end]
            
            # Optimize on training data
            logger.info(f"  Optimizing on {len(train_data)} training bars...")
            optimal_params = self.optimize_window(
                train_data,
                param_space,
                objective_func,
                n_trials,
                algorithm
            )
            window.optimal_params = optimal_params
            
            # Calculate in-sample metrics
            logger.info("  Calculating in-sample metrics...")
            is_equity, is_trades = strategy_func(train_data, optimal_params)
            window.is_metrics = calculate_all_metrics(is_equity, is_trades)
            
            # Calculate out-of-sample metrics
            logger.info(f"  Testing on {len(test_data)} out-of-sample bars...")
            oos_equity, oos_trades = strategy_func(test_data, optimal_params)
            window.oos_metrics = calculate_all_metrics(oos_equity, oos_trades)
            
            # Log results
            logger.info(f"  IS Sharpe: {window.is_metrics.sharpe_ratio:.2f}, "
                       f"OOS Sharpe: {window.oos_metrics.sharpe_ratio:.2f}")
        
        # Aggregate out-of-sample results
        combined_oos_equity, combined_oos_trades = self._combine_oos_results(df, windows, strategy_func)
        combined_oos_metrics = calculate_all_metrics(combined_oos_equity, combined_oos_trades)
        
        # Calculate parameter stability
        param_history = [w.optimal_params for w in windows]
        param_stability = calculate_parameter_stability(param_history)
        
        # Calculate average WFE
        wfes = []
        for window in windows:
            if window.is_metrics.total_return != 0:
                wfe = window.oos_metrics.total_return / window.is_metrics.total_return
                wfes.append(wfe)
        average_wfe = np.mean(wfes) if wfes else 0.0
        
        # Calculate total time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        results = WalkForwardResults(
            windows=windows,
            combined_oos_metrics=combined_oos_metrics,
            parameter_stability=param_stability,
            average_wfe=average_wfe,
            total_windows=len(windows),
            optimization_time=optimization_time
        )
        
        logger.info(f"Walk-forward optimization complete in {optimization_time:.1f}s")
        logger.info(f"Combined OOS Sharpe: {combined_oos_metrics.sharpe_ratio:.2f}")
        logger.info(f"Average WFE: {average_wfe:.2f}")
        
        return results
    
    def _combine_oos_results(self,
                            df: pd.DataFrame,
                            windows: List[WalkForwardWindow],
                            strategy_func: Callable) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Combine out-of-sample results from all windows
        
        Args:
            df: Full dataset
            windows: List of optimized windows
            strategy_func: Strategy function
            
        Returns:
            Tuple of (combined equity curve, combined trades)
        """
        all_equity = []
        all_trades = []
        
        for window in windows:
            test_data = df.loc[window.test_start:window.test_end]
            equity, trades = strategy_func(test_data, window.optimal_params)
            
            all_equity.append(equity)
            if trades is not None and len(trades) > 0:
                all_trades.append(trades)
        
        # Combine equity curves
        combined_equity = pd.concat(all_equity)
        
        # Combine trades
        combined_trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
        
        return combined_equity, combined_trades
    
    def save_results(self, results: WalkForwardResults, output_dir: str):
        """
        Save walk-forward results to disk
        
        Args:
            results: WalkForwardResults object
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            'total_windows': results.total_windows,
            'average_wfe': results.average_wfe,
            'optimization_time': results.optimization_time,
            'combined_oos_metrics': metrics_to_dict(results.combined_oos_metrics),
            'parameter_stability': results.parameter_stability
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed window results
        window_data = []
        for window in results.windows:
            window_dict = {
                'window_id': window.window_id,
                'train_start': str(window.train_start),
                'train_end': str(window.train_end),
                'test_start': str(window.test_start),
                'test_end': str(window.test_end),
                'optimal_params': window.optimal_params,
                'is_metrics': metrics_to_dict(window.is_metrics),
                'oos_metrics': metrics_to_dict(window.oos_metrics)
            }
            window_data.append(window_dict)
        
        df_windows = pd.DataFrame(window_data)
        df_windows.to_csv(output_path / 'windows.csv', index=False)
        
        # Save parameter evolution
        param_history = pd.DataFrame([w.optimal_params for w in results.windows])
        param_history['window_id'] = [w.window_id for w in results.windows]
        param_history.to_csv(output_path / 'parameter_evolution.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")
