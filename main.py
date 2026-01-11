"""
QTAlgo Super26 Strategy - Simple Backtesting Script

This script runs the QTAlgo Super26 trading strategy on historical OHLCV data
and outputs key performance metrics.
"""

import argparse
import yaml
import logging
from pathlib import Path

from src.data.loader import DataLoader
from src.backtest import Backtester
from src.metrics import calculate_metrics, print_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/strategy_params.yaml'):
    """
    Load strategy parameters from YAML file.
    
    Args:
        config_path: Path to strategy configuration file
        
    Returns:
        Dictionary with flattened strategy parameters
    """
    with open(config_path, 'r') as f:
        strategy_config = yaml.safe_load(f)
    
    # Flatten nested configuration
    params = {}
    for section, values in strategy_config.items():
        if isinstance(values, dict):
            params.update(values)
    
    return params


def main():
    """Main backtest execution function"""
    parser = argparse.ArgumentParser(
        description='QTAlgo Super26 Strategy Backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --data data/SPY_5y.csv
  python main.py --data data/SPY_5y.csv --config config/strategy_params.yaml
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/ohlcv_data.csv',
        help='Path to CSV file with OHLCV data (default: data/ohlcv_data.csv)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/strategy_params.yaml',
        help='Path to strategy configuration file (default: config/strategy_params.yaml)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000,
        help='Initial capital for backtest (default: 100000)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        params = load_config(args.config)
        
        # Load data
        logger.info(f"Loading data from {args.data}")
        loader = DataLoader()
        df = loader.load_csv(args.data)
        
        # Validate data
        if not loader.validate_data(df):
            logger.error("Data validation failed")
            return
        
        # Handle any missing data
        df = loader.handle_missing_data(df)
        
        logger.info(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Run backtest
        logger.info("Running backtest...")
        backtester = Backtester(initial_capital=args.initial_capital)
        equity_curve, trades_df = backtester.run(df, params)
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = calculate_metrics(equity_curve, trades_df, args.initial_capital)
        
        # Print results
        print_metrics(metrics)
        
        # Save trades to CSV if there are any
        if len(trades_df) > 0:
            output_path = 'backtest_trades.csv'
            trades_df.to_csv(output_path, index=False)
            logger.info(f"Trades saved to {output_path}")
        
        # Save equity curve to CSV
        equity_path = 'backtest_equity.csv'
        equity_curve.to_csv(equity_path, header=['equity'])
        logger.info(f"Equity curve saved to {equity_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please ensure your data file exists at the specified path")
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)


if __name__ == "__main__":
    main()
