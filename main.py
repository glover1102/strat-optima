"""
Main Entry Point for QTAlgo Super26 Strategy Optimization

Provides CLI and API interfaces for running walk-forward optimization.
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.strategy.indicators import calculate_all_indicators
from src.strategy.signals import generate_signals
from src.strategy.exits import simulate_exits
from src.data.loader import DataLoader
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import ParameterSpace, load_base_parameters, merge_parameters
from src.optimization.metrics import calculate_all_metrics
from src.utils.plotting import (
    plot_strategy_performance, plot_walk_forward_efficiency,
    plot_parameter_stability, save_figure
)
from src.utils.reporting import (
    generate_performance_report, generate_walk_forward_report,
    save_report_to_file, export_trades_to_csv
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def strategy_function(df, params):
    """
    Complete strategy function that takes data and parameters,
    returns trades DataFrame.
    
    Args:
        df: OHLCV DataFrame
        params: Strategy parameters
        
    Returns:
        DataFrame with trade results
    """
    # Calculate indicators
    df_with_indicators = calculate_all_indicators(df, params)
    
    # Generate signals
    df_with_signals = generate_signals(df_with_indicators, params)
    
    # Simulate exits and get trades
    trades_df = simulate_exits(df, params, df_with_signals)
    
    return trades_df


def run_single_backtest(config_path: str, data_path: str, symbol: str,
                       output_dir: str = "results/") -> None:
    """
    Run a single backtest with default parameters.
    
    Args:
        config_path: Path to strategy config
        data_path: Path to data file
        symbol: Symbol name
        output_dir: Output directory for results
    """
    logger.info("Running single backtest")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base parameters
    base_params = load_base_parameters(config_path)
    
    # Load data
    loader = DataLoader()
    df = loader.load_csv(data_path, symbol)
    df = loader.clean_data(df)
    
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Run strategy
    trades_df = strategy_function(df, base_params)
    
    # Calculate metrics
    metrics = calculate_all_metrics(trades_df)
    
    # Generate report
    report = generate_performance_report(metrics, trades_df, 
                                        title=f"{symbol} Backtest Results")
    print(report)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_report_to_file(report, output_path / f"{symbol}_report.txt")
    export_trades_to_csv(trades_df, output_path / f"{symbol}_trades.csv")
    
    logger.info(f"Results saved to {output_dir}")


def run_walk_forward_optimization(strategy_config_path: str,
                                  optimization_config_path: str,
                                  data_path: str,
                                  symbol: str,
                                  output_dir: str = "results/") -> None:
    """
    Run walk-forward optimization.
    
    Args:
        strategy_config_path: Path to strategy parameters config
        optimization_config_path: Path to optimization config
        data_path: Path to data file
        symbol: Symbol name
        output_dir: Output directory for results
    """
    logger.info("Starting walk-forward optimization")
    
    # Load configurations
    with open(strategy_config_path, 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    with open(optimization_config_path, 'r') as f:
        opt_config = yaml.safe_load(f)
    
    # Load data
    loader = DataLoader()
    df = loader.load_csv(data_path, symbol)
    df = loader.clean_data(df)
    
    logger.info(f"Loaded {len(df)} rows of data from {df.index[0]} to {df.index[-1]}")
    
    # Setup parameter space
    param_space = ParameterSpace(strategy_config_path)
    base_params = load_base_parameters(strategy_config_path)
    
    logger.info(f"Optimizing {len(param_space.parameters)} parameters")
    
    # Create wrapped strategy function
    def wrapped_strategy(data, opt_params):
        merged_params = merge_parameters(base_params, opt_params)
        return strategy_function(data, merged_params)
    
    # Create optimizer
    wf_config = opt_config.get('walk_forward', {})
    wf_config.update(opt_config.get('optimization', {}))
    
    optimizer = WalkForwardOptimizer(wf_config)
    
    # Run optimization
    periods = optimizer.run_walk_forward(df, param_space, wrapped_strategy)
    
    # Calculate results
    wfe = optimizer.calculate_wfe()
    param_stability = optimizer.analyze_parameter_stability()
    aggregate_results = optimizer.get_aggregate_results()
    
    logger.info(f"Walk-forward efficiency: {wfe:.2f}")
    logger.info(f"Average OOS Sharpe: {aggregate_results['out_of_sample']['avg_sharpe']:.2f}")
    
    # Generate reports
    wf_report = generate_walk_forward_report(periods, wfe, param_stability)
    print(wf_report)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save text report
    save_report_to_file(wf_report, output_path / f"{symbol}_wf_report_{timestamp}.txt")
    
    # Save parameter stability
    if param_stability is not None and not param_stability.empty:
        param_stability.to_csv(output_path / f"{symbol}_param_stability_{timestamp}.csv")
    
    # Save optimizer state
    optimizer.save_results(output_path / f"{symbol}_wf_results_{timestamp}.pkl")
    
    # Generate plots
    logger.info("Generating visualizations")
    
    # Walk-forward efficiency plot
    wf_fig = plot_walk_forward_efficiency(periods, 
                                          title=f"{symbol} Walk-Forward Analysis")
    save_figure(wf_fig, output_path / f"{symbol}_wf_analysis_{timestamp}.html")
    
    # Parameter stability plot
    if param_stability is not None and not param_stability.empty:
        # Create time series of parameters
        params_over_time = []
        for period in periods:
            if period.best_params:
                params_over_time.append(period.best_params)
        
        if params_over_time:
            import pandas as pd
            params_df = pd.DataFrame(params_over_time)
            param_fig = plot_parameter_stability(params_df,
                                                title=f"{symbol} Parameter Evolution")
            save_figure(param_fig, output_path / f"{symbol}_param_evolution_{timestamp}.html")
    
    logger.info(f"Optimization complete. Results saved to {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='QTAlgo Super26 Strategy Walk-Forward Optimization'
    )
    
    parser.add_argument(
        'mode',
        choices=['backtest', 'optimize'],
        help='Run mode: backtest or optimize'
    )
    
    parser.add_argument(
        '--strategy-config',
        default='config/strategy_params.yaml',
        help='Path to strategy parameters config'
    )
    
    parser.add_argument(
        '--opt-config',
        default='config/optimization_config.yaml',
        help='Path to optimization config (for optimize mode)'
    )
    
    parser.add_argument(
        '--data',
        required=True,
        help='Path to data file (CSV)'
    )
    
    parser.add_argument(
        '--symbol',
        default='BTCUSD',
        help='Symbol name'
    )
    
    parser.add_argument(
        '--output',
        default='results/',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'backtest':
            run_single_backtest(
                args.strategy_config,
                args.data,
                args.symbol,
                args.output
            )
        elif args.mode == 'optimize':
            run_walk_forward_optimization(
                args.strategy_config,
                args.opt_config,
                args.data,
                args.symbol,
                args.output
            )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
