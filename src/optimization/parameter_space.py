"""
Parameter Space Definitions for Walk-Forward Optimization

Defines the parameter search space and manages parameter bounds,
constraints, and sampling strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class ParameterDefinition:
    """Definition of a single parameter to optimize."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    param_type: str = "float"  # "float", "int", "categorical"
    categories: Optional[List[Any]] = None
    default: Optional[float] = None


class ParameterSpace:
    """
    Manages the parameter search space for optimization.
    
    Provides functionality to:
    - Define parameter bounds and types
    - Sample parameters for optimization
    - Validate parameter values
    - Load/save parameter configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize parameter space.
        
        Args:
            config_path: Path to strategy parameters YAML file
        """
        self.parameters: Dict[str, ParameterDefinition] = {}
        
        if config_path:
            self.load_from_config(config_path)
    
    def add_parameter(self, name: str, min_value: float, max_value: float,
                     step: Optional[float] = None, param_type: str = "float",
                     default: Optional[float] = None) -> None:
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            step: Step size for grid search (optional)
            param_type: Type of parameter ("float", "int", "categorical")
            default: Default value
        """
        param = ParameterDefinition(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step=step,
            param_type=param_type,
            default=default
        )
        self.parameters[name] = param
    
    def load_from_config(self, config_path: str) -> None:
        """
        Load parameter space from YAML configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load optimization ranges
        if 'optimization_ranges' in config:
            for param_name, range_def in config['optimization_ranges'].items():
                if isinstance(range_def, list) and len(range_def) == 3:
                    min_val, max_val, step = range_def
                    
                    # Determine parameter type
                    param_type = "int" if isinstance(step, int) and step >= 1 else "float"
                    
                    # Get default from config
                    default = self._get_default_from_config(config, param_name)
                    
                    self.add_parameter(
                        name=param_name,
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        param_type=param_type,
                        default=default
                    )
    
    def _get_default_from_config(self, config: Dict, param_name: str) -> Optional[float]:
        """Extract default value from config structure."""
        # Try different config sections
        sections = ['entry', 'risk', 'adx', 'weights']
        
        for section in sections:
            if section in config and param_name in config[section]:
                return config[section][param_name]
        
        # Try nested parameter names (e.g., w_adx in weights)
        if param_name.startswith('w_'):
            if 'weights' in config and param_name in config['weights']:
                return config['weights'][param_name]
        
        return None
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get parameter bounds for optimization algorithms.
        
        Returns:
            List of (min, max) tuples for each parameter
        """
        return [(p.min_value, p.max_value) for p in self.parameters.values()]
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, float]]:
        """
        Generate random parameter samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, param in self.parameters.items():
                if param.param_type == "int":
                    value = np.random.randint(
                        int(param.min_value),
                        int(param.max_value) + 1
                    )
                else:
                    value = np.random.uniform(param.min_value, param.max_value)
                
                sample[name] = value
            
            samples.append(sample)
        
        return samples
    
    def generate_grid(self) -> List[Dict[str, float]]:
        """
        Generate grid of parameter combinations.
        
        Uses step size to create grid. Warning: Can be very large!
        
        Returns:
            List of parameter dictionaries
        """
        param_grids = []
        param_names = []
        
        for name, param in self.parameters.items():
            if param.step is not None:
                if param.param_type == "int":
                    grid = np.arange(
                        int(param.min_value),
                        int(param.max_value) + 1,
                        int(param.step)
                    )
                else:
                    grid = np.arange(
                        param.min_value,
                        param.max_value + param.step / 2,
                        param.step
                    )
                param_grids.append(grid)
                param_names.append(name)
            else:
                # Default to 10 points if no step defined
                grid = np.linspace(param.min_value, param.max_value, 10)
                param_grids.append(grid)
                param_names.append(name)
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_grids))
        
        # Convert to list of dictionaries
        samples = []
        for combo in combinations:
            sample = dict(zip(param_names, combo))
            samples.append(sample)
        
        return samples
    
    def validate_parameters(self, params: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate parameter values.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for name, value in params.items():
            if name not in self.parameters:
                issues.append(f"Unknown parameter: {name}")
                continue
            
            param_def = self.parameters[name]
            
            # Check bounds
            if value < param_def.min_value or value > param_def.max_value:
                issues.append(
                    f"{name} = {value} is out of bounds "
                    f"[{param_def.min_value}, {param_def.max_value}]"
                )
            
            # Check type
            if param_def.param_type == "int" and not isinstance(value, (int, np.integer)):
                if not value.is_integer():
                    issues.append(f"{name} must be an integer")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clip_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Clip parameters to valid bounds.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Clipped parameter dictionary
        """
        clipped = {}
        
        for name, value in params.items():
            if name in self.parameters:
                param_def = self.parameters[name]
                clipped_value = np.clip(value, param_def.min_value, param_def.max_value)
                
                if param_def.param_type == "int":
                    clipped_value = int(round(clipped_value))
                
                clipped[name] = clipped_value
            else:
                clipped[name] = value
        
        return clipped
    
    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values.
        
        Returns:
            Dictionary of default parameters
        """
        defaults = {}
        
        for name, param in self.parameters.items():
            if param.default is not None:
                defaults[name] = param.default
            else:
                # Use midpoint if no default specified
                mid = (param.min_value + param.max_value) / 2
                if param.param_type == "int":
                    mid = int(round(mid))
                defaults[name] = mid
        
        return defaults
    
    def create_optuna_space(self, trial) -> Dict[str, float]:
        """
        Create Optuna trial suggestions for parameter space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        for name, param in self.parameters.items():
            if param.param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(param.min_value),
                    int(param.max_value),
                    step=int(param.step) if param.step else 1
                )
            elif param.param_type == "float":
                if param.step:
                    params[name] = trial.suggest_discrete_uniform(
                        name,
                        param.min_value,
                        param.max_value,
                        param.step
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        param.min_value,
                        param.max_value
                    )
            elif param.param_type == "categorical" and param.categories:
                params[name] = trial.suggest_categorical(name, param.categories)
        
        return params
    
    def get_parameter_info(self) -> pd.DataFrame:
        """
        Get parameter information as DataFrame.
        
        Returns:
            DataFrame with parameter definitions
        """
        import pandas as pd
        
        info = []
        for name, param in self.parameters.items():
            info.append({
                'name': name,
                'type': param.param_type,
                'min': param.min_value,
                'max': param.max_value,
                'step': param.step,
                'default': param.default
            })
        
        return pd.DataFrame(info)


def load_base_parameters(config_path: str) -> Dict[str, Any]:
    """
    Load base strategy parameters from config file.
    
    These are the parameters not being optimized.
    
    Args:
        config_path: Path to strategy configuration YAML
        
    Returns:
        Dictionary of base parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_params = {}
    
    # Flatten config structure
    for section_name, section in config.items():
        if isinstance(section, dict) and section_name != 'optimization_ranges':
            for key, value in section.items():
                # Create prefixed key
                param_key = f"{section_name}_{key}"
                base_params[param_key] = value
                
                # Also add unprefixed for common params
                if section_name in ['entry', 'risk']:
                    base_params[key] = value
    
    return base_params


def merge_parameters(base_params: Dict, opt_params: Dict) -> Dict:
    """
    Merge base parameters with optimized parameters.
    
    Optimized parameters override base parameters.
    
    Args:
        base_params: Base parameter dictionary
        opt_params: Optimized parameter dictionary
        
    Returns:
        Merged parameter dictionary
    """
    merged = base_params.copy()
    merged.update(opt_params)
    return merged
