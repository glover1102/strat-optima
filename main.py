"""
Main Entry Point for Walk-Forward Optimization Framework

Provides FastAPI web interface and CLI for running optimizations.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from src.data.loader import DataLoader
from src.strategy.indicators import calculate_all_indicators
from src.strategy.signals import generate_all_signals
from src.strategy.exits import ExitManager, Position
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import ParameterSpace
from src.optimization.metrics import calculate_all_metrics
from src.utils.reporting import print_summary_report, export_full_report
from src.utils.plotting import create_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="QTAlgo Super26 Walk-Forward Optimization",
    description="Walk-forward optimization framework for algorithmic trading strategies",
    version="1.0.0"
)

# Store optimization status
optimization_status = {
    'running': False,
    'progress': 0,
    'message': '',
    'results': None
}


class OptimizationRequest(BaseModel):
    """Request model for optimization"""
    symbols: list[str] = ['SPY']
    start_date: str = '2018-01-01'
    end_date: Optional[str] = None
    timeframe: str = '1d'
    n_trials: int = 100
    algorithm: str = 'optuna'


def run_backtest(df, params):
    """
    Run backtest with given parameters
    
    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters
        
    Returns:
        Tuple of (equity_curve, trades_df)
    """
    # Calculate indicators
    df_with_indicators = calculate_all_indicators(df.copy(), params)
    
    # Generate signals
    df_with_signals = generate_all_signals(df_with_indicators, params)
    
    # Simple equity curve calculation (for demonstration)
    # In production, this should be more sophisticated
    initial_capital = 100000
    equity = [initial_capital]
    trades = []
    position = None
    
    for i in range(1, len(df_with_signals)):
        current_equity = equity[-1]
        
        # Check for entry signal
        if position is None and df_with_signals['entry_signal'].iloc[i] != 0:
            position = Position(
                entry_price=df_with_signals['close'].iloc[i],
                entry_bar=i,
                direction=int(df_with_signals['entry_signal'].iloc[i]),
                size=1.0
            )
        
        # Check for exit if in position
        if position is not None:
            exit_manager = ExitManager(params)
            position = exit_manager.calculate_exit_levels(position)
            position = exit_manager.update_trailing_stop(position, df_with_signals['close'].iloc[i])
            
            should_exit, exit_reason, exit_price = exit_manager.check_exit(
                position, 
                df_with_signals.iloc[i]
            )
            
            if should_exit:
                # Calculate PnL
                if position.direction == 1:
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - exit_price) / position.entry_price
                
                pnl = current_equity * pnl_pct
                current_equity += pnl
                
                trades.append({
                    'entry_date': df_with_signals.index[position.entry_bar],
                    'exit_date': df_with_signals.index[i],
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'direction': position.direction,
                    'pnl': pnl,
                    'pnl_percent': pnl_pct * 100,
                    'exit_reason': exit_reason
                })
                
                position = None
        
        equity.append(current_equity)
    
    # Convert to pandas objects
    import pandas as pd
    equity_curve = pd.Series(equity, index=df_with_signals.index[:len(equity)])
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return equity_curve, trades_df


def objective_function(df, params):
    """Objective function for optimization"""
    equity, trades = run_backtest(df, params)
    
    if len(equity) < 2:
        return -np.inf
    
    metrics = calculate_all_metrics(equity, trades)
    
    # Multi-objective: combine Sharpe ratio, max drawdown, and win rate
    # Normalize and weight
    sharpe_score = metrics.sharpe_ratio
    dd_score = -metrics.max_drawdown / 100  # Negative because lower is better
    wr_score = (metrics.win_rate - 0.5) * 10  # Scale around 50%
    
    # Weighted combination
    score = (0.5 * sharpe_score) + (0.3 * dd_score) + (0.2 * wr_score)
    
    return score


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "QTAlgo Super26 Walk-Forward Optimization API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/status")
async def get_status():
    """Get optimization status"""
    return optimization_status


@app.post("/optimize")
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start walk-forward optimization"""
    global optimization_status
    
    if optimization_status['running']:
        raise HTTPException(status_code=400, detail="Optimization already running")
    
    # Reset status
    optimization_status = {
        'running': True,
        'progress': 0,
        'message': 'Starting optimization...',
        'results': None
    }
    
    # Run optimization in background
    background_tasks.add_task(
        run_optimization_task,
        request.symbols,
        request.start_date,
        request.end_date,
        request.timeframe,
        request.n_trials,
        request.algorithm
    )
    
    return {"message": "Optimization started", "status": optimization_status}


def run_optimization_task(symbols, start_date, end_date, timeframe, n_trials, algorithm):
    """Background task for optimization"""
    global optimization_status
    
    try:
        # Load configuration
        with open('config/optimization_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        with open('config/strategy_params.yaml', 'r') as f:
            strategy_config = yaml.safe_load(f)
        
        # Flatten strategy config
        params = {}
        for section, values in strategy_config.items():
            if isinstance(values, dict):
                params.update(values)
        
        # Load data
        optimization_status['message'] = f'Loading data for {symbols}...'
        loader = DataLoader(source='yfinance')
        
        all_results = {}
        
        for i, symbol in enumerate(symbols):
            optimization_status['message'] = f'Processing {symbol} ({i+1}/{len(symbols)})...'
            optimization_status['progress'] = int((i / len(symbols)) * 100)
            
            # Load symbol data
            df = loader.load_data(symbol, start_date, end_date, timeframe)
            
            if not loader.validate_data(df):
                logger.warning(f"Data validation failed for {symbol}")
                continue
            
            df = loader.handle_missing_data(df)
            
            # Create parameter space
            param_space = ParameterSpace(config['parameter_bounds'])
            
            # Initialize optimizer
            optimizer = WalkForwardOptimizer(
                window_type=config['walk_forward']['window_type'],
                train_period_months=config['walk_forward']['train_period_months'],
                test_period_months=config['walk_forward']['test_period_months'],
                step_size_months=config['walk_forward']['step_size_months']
            )
            
            # Run optimization
            optimization_status['message'] = f'Optimizing {symbol}...'
            results = optimizer.run_walk_forward(
                df,
                param_space,
                run_backtest,
                objective_function,
                n_trials,
                algorithm
            )
            
            # Save results
            output_dir = f'results/{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            optimizer.save_results(results, output_dir)
            export_full_report(results, output_dir, format='both')
            
            # Create dashboard
            combined_equity, _ = optimizer._combine_oos_results(df, results.windows, run_backtest)
            param_history = pd.DataFrame([w.optimal_params for w in results.windows])
            param_history['window_id'] = [w.window_id for w in results.windows]
            create_dashboard(results, combined_equity, param_history, output_dir + '/plots')
            
            all_results[symbol] = {
                'sharpe_ratio': results.combined_oos_metrics.sharpe_ratio,
                'total_return': results.combined_oos_metrics.total_return,
                'max_drawdown': results.combined_oos_metrics.max_drawdown,
                'win_rate': results.combined_oos_metrics.win_rate,
                'output_dir': output_dir
            }
            
            # Print summary
            print_summary_report(results)
        
        optimization_status['running'] = False
        optimization_status['progress'] = 100
        optimization_status['message'] = 'Optimization complete'
        optimization_status['results'] = all_results
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        optimization_status['running'] = False
        optimization_status['message'] = f'Error: {str(e)}'


@app.get("/results")
async def get_results():
    """Get optimization results"""
    if optimization_status['results'] is None:
        raise HTTPException(status_code=404, detail="No results available")
    
    return optimization_status['results']


def cli_main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='QTAlgo Super26 Walk-Forward Optimization')
    
    parser.add_argument('--symbols', nargs='+', default=['SPY'], help='Symbols to optimize')
    parser.add_argument('--start-date', type=str, default='2018-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (1d, 1h, etc.)')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--algorithm', type=str, default='optuna', help='Optimization algorithm')
    parser.add_argument('--mode', type=str, default='cli', choices=['cli', 'server'], 
                       help='Run mode: cli or server')
    parser.add_argument('--port', type=int, default=8000, help='Server port (server mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        # Run FastAPI server
        logger.info(f"Starting server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Run CLI optimization
        logger.info("Starting CLI optimization")
        run_optimization_task(
            args.symbols,
            args.start_date,
            args.end_date,
            args.timeframe,
            args.n_trials,
            args.algorithm
        )


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    cli_main()
