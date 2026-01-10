"""
Parameter Space Definition Module

Defines the parameter space for optimization with bounds and constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ParameterDefinition:
    """Definition of a single parameter"""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    param_type: str = 'continuous'  # 'continuous', 'integer', 'categorical'
    default: Optional[float] = None


class ParameterSpace:
    """Manages parameter space for optimization"""
    
    def __init__(self, config: Dict[str, List]):
        """
        Initialize parameter space
        
        Args:
            config: Dictionary mapping parameter names to [min, max, step] or [min, max]
        """
        self.parameters = {}
        self._parse_config(config)
    
    def _parse_config(self, config: Dict[str, List]):
        """Parse configuration and create parameter definitions"""
        for name, bounds in config.items():
            if len(bounds) == 3:
                # [min, max, step]
                min_val, max_val, step = bounds
                param_type = 'integer' if step == int(step) and step >= 1 else 'continuous'
            elif len(bounds) == 2:
                # [min, max] - continuous
                min_val, max_val = bounds
                step = None
                param_type = 'continuous'
            else:
                raise ValueError(f"Invalid bounds for parameter {name}: {bounds}")
            
            self.parameters[name] = ParameterDefinition(
                name=name,
                min_value=min_val,
                max_value=max_val,
                step=step,
                param_type=param_type
            )
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get parameter bounds for optimization
        
        Returns:
            List of (min, max) tuples
        """
        return [(p.min_value, p.max_value) for p in self.parameters.values()]
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names"""
        return list(self.parameters.keys())
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, float]]:
        """
        Generate random parameter samples
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, param in self.parameters.items():
                if param.param_type == 'integer':
                    value = np.random.randint(int(param.min_value), int(param.max_value) + 1)
                else:
                    value = np.random.uniform(param.min_value, param.max_value)
                    if param.step is not None:
                        # Round to nearest step
                        value = round(value / param.step) * param.step
                
                sample[name] = value
            
            samples.append(sample)
        
        return samples
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """
        Validate parameter values are within bounds
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            True if valid
        """
        for name, value in params.items():
            if name not in self.parameters:
                return False
            
            param = self.parameters[name]
            if value < param.min_value or value > param.max_value:
                return False
        
        return True
    
    def clip_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Clip parameter values to bounds
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Clipped parameters
        """
        clipped = {}
        for name, value in params.items():
            if name in self.parameters:
                param = self.parameters[name]
                clipped[name] = np.clip(value, param.min_value, param.max_value)
                
                if param.step is not None:
                    clipped[name] = round(clipped[name] / param.step) * param.step
            else:
                clipped[name] = value
        
        return clipped
    
    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values (midpoint of range)
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {}
        for name, param in self.parameters.items():
            if param.default is not None:
                defaults[name] = param.default
            else:
                # Use midpoint
                midpoint = (param.min_value + param.max_value) / 2
                if param.step is not None:
                    midpoint = round(midpoint / param.step) * param.step
                defaults[name] = midpoint
        
        return defaults
    
    def get_grid_points(self, n_points: int = 10) -> List[Dict[str, float]]:
        """
        Generate grid points for grid search
        
        Args:
            n_points: Number of points per parameter
            
        Returns:
            List of parameter combinations
        """
        from itertools import product
        
        # Generate points for each parameter
        param_points = {}
        for name, param in self.parameters.items():
            if param.step is not None:
                # Use step size
                points = np.arange(param.min_value, param.max_value + param.step, param.step)
            else:
                # Use n_points
                points = np.linspace(param.min_value, param.max_value, n_points)
            
            param_points[name] = points
        
        # Generate all combinations
        names = list(param_points.keys())
        combinations = product(*[param_points[name] for name in names])
        
        grid = []
        for combo in combinations:
            grid.append(dict(zip(names, combo)))
        
        return grid


def create_parameter_space_from_config(config_path: str) -> ParameterSpace:
    """
    Create parameter space from YAML configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ParameterSpace object
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    param_bounds = config.get('parameter_bounds', {})
    return ParameterSpace(param_bounds)


def calculate_parameter_stability(param_history: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate parameter stability across optimization windows
    
    Args:
        param_history: List of parameter dictionaries from each window
        
    Returns:
        Dictionary mapping parameter names to stability scores (0-1)
    """
    if len(param_history) < 2:
        return {}
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(param_history)
    
    stability = {}
    for col in df.columns:
        # Calculate coefficient of variation (lower is more stable)
        mean = df[col].mean()
        std = df[col].std()
        
        if mean == 0:
            stability[col] = 0.0
        else:
            cv = std / abs(mean)
            # Convert to 0-1 scale (1 = most stable)
            stability[col] = 1.0 / (1.0 + cv)
    
    return stability


def detect_parameter_drift(param_history: List[Dict[str, float]], 
                          threshold: float = 0.3) -> Dict[str, bool]:
    """
    Detect if parameters are drifting over time
    
    Args:
        param_history: List of parameter dictionaries
        threshold: Threshold for drift detection (as percentage of range)
        
    Returns:
        Dictionary indicating if each parameter is drifting
    """
    if len(param_history) < 3:
        return {}
    
    import pandas as pd
    df = pd.DataFrame(param_history)
    
    drift_detected = {}
    for col in df.columns:
        # Calculate trend
        x = np.arange(len(df))
        y = df[col].values
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by range
        value_range = y.max() - y.min()
        if value_range == 0:
            drift_detected[col] = False
        else:
            normalized_slope = abs(slope) / value_range
            drift_detected[col] = normalized_slope > threshold
    
    return drift_detected


def suggest_parameter_ranges(param_history: List[Dict[str, float]], 
                            expansion_factor: float = 1.5) -> Dict[str, Tuple[float, float]]:
    """
    Suggest new parameter ranges based on historical optimal values
    
    Args:
        param_history: List of parameter dictionaries
        expansion_factor: Factor to expand range around observed values
        
    Returns:
        Dictionary mapping parameter names to suggested (min, max) ranges
    """
    if len(param_history) < 2:
        return {}
    
    import pandas as pd
    df = pd.DataFrame(param_history)
    
    suggestions = {}
    for col in df.columns:
        values = df[col].values
        mean = values.mean()
        std = values.std()
        
        # Suggest range: mean +/- expansion_factor * std
        min_val = mean - expansion_factor * std
        max_val = mean + expansion_factor * std
        
        suggestions[col] = (min_val, max_val)
    
    return suggestions
