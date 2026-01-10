"""
FastAPI Web Interface for QTAlgo Super26 Strategy

Provides REST API endpoints for running optimizations and retrieving results.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
import logging
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.strategy.indicators import calculate_all_indicators
from src.strategy.signals import generate_signals
from src.strategy.exits import simulate_exits
from src.data.loader import DataLoader
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import ParameterSpace, load_base_parameters, merge_parameters
from src.optimization.metrics import calculate_all_metrics, metrics_to_dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="QTAlgo Super26 API",
    description="Walk-Forward Optimization API for QTAlgo Super26 Strategy",
    version="1.0.0"
)

# Global state for tracking jobs
optimization_jobs = {}


# Request/Response Models
class BacktestRequest(BaseModel):
    symbol: str
    data_path: Optional[str] = None
    parameters: Optional[Dict] = None


class OptimizationRequest(BaseModel):
    symbol: str
    data_path: str
    mode: str = "rolling"
    train_months: int = 12
    test_months: int = 3
    n_trials: int = 100
    algorithm: str = "optuna"


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None


# Helper Functions
def strategy_function(df, params):
    """Complete strategy function."""
    df_with_indicators = calculate_all_indicators(df, params)
    df_with_signals = generate_signals(df_with_indicators, params)
    trades_df = simulate_exits(df, params, df_with_signals)
    return trades_df


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "QTAlgo Super26 API",
        "version": "1.0.0",
        "endpoints": {
            "backtest": "/api/backtest",
            "optimize": "/api/optimize",
            "status": "/api/status/{job_id}",
            "results": "/api/results/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run a single backtest with given parameters.
    
    Returns performance metrics.
    """
    try:
        logger.info(f"Running backtest for {request.symbol}")
        
        # Load configuration
        config_path = 'config/strategy_params.yaml'
        base_params = load_base_parameters(config_path)
        
        # Override with custom parameters if provided
        if request.parameters:
            base_params.update(request.parameters)
        
        # Load data
        loader = DataLoader()
        if request.data_path:
            df = loader.load_csv(request.data_path, request.symbol)
        else:
            # Try default path
            df = loader.load_csv(f'data/raw/{request.symbol}.csv', request.symbol)
        
        df = loader.clean_data(df)
        
        # Run strategy
        trades_df = strategy_function(df, base_params)
        
        # Calculate metrics
        metrics = calculate_all_metrics(trades_df)
        
        # Return results
        return {
            "symbol": request.symbol,
            "metrics": metrics_to_dict(metrics),
            "total_trades": len(trades_df),
            "date_range": {
                "start": str(df.index[0]),
                "end": str(df.index[-1])
            }
        }
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def run_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start walk-forward optimization in background.
    
    Returns job ID for tracking progress.
    """
    try:
        # Generate job ID
        job_id = f"{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize job status
        optimization_jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Optimization queued"
        }
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimization_task,
            job_id,
            request
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Optimization started"
        }
    
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_optimization_task(job_id: str, request: OptimizationRequest):
    """Background task for running optimization."""
    try:
        # Update status
        optimization_jobs[job_id] = {
            "status": "running",
            "progress": 10,
            "message": "Loading data and configuration"
        }
        
        # Load configurations
        strategy_config_path = 'config/strategy_params.yaml'
        with open(strategy_config_path, 'r') as f:
            strategy_config = yaml.safe_load(f)
        
        # Load data
        loader = DataLoader()
        df = loader.load_csv(request.data_path, request.symbol)
        df = loader.clean_data(df)
        
        optimization_jobs[job_id]["progress"] = 20
        optimization_jobs[job_id]["message"] = "Setting up parameter space"
        
        # Setup parameter space
        param_space = ParameterSpace(strategy_config_path)
        base_params = load_base_parameters(strategy_config_path)
        
        # Create wrapped strategy function
        def wrapped_strategy(data, opt_params):
            merged_params = merge_parameters(base_params, opt_params)
            return strategy_function(data, merged_params)
        
        optimization_jobs[job_id]["progress"] = 30
        optimization_jobs[job_id]["message"] = "Running walk-forward optimization"
        
        # Create optimizer
        wf_config = {
            'mode': request.mode,
            'train_period_months': request.train_months,
            'test_period_months': request.test_months,
            'step_months': 3,
            'algorithm': request.algorithm,
            'n_trials': request.n_trials,
            'n_jobs': -1,
            'objectives': {
                'sharpe_ratio': {'weight': 0.4, 'direction': 'maximize'},
                'max_drawdown': {'weight': 0.3, 'direction': 'minimize'},
                'win_rate': {'weight': 0.2, 'direction': 'maximize'},
                'profit_factor': {'weight': 0.1, 'direction': 'maximize'}
            }
        }
        
        optimizer = WalkForwardOptimizer(wf_config)
        
        # Run optimization
        periods = optimizer.run_walk_forward(df, param_space, wrapped_strategy)
        
        optimization_jobs[job_id]["progress"] = 80
        optimization_jobs[job_id]["message"] = "Calculating results"
        
        # Calculate results
        wfe = optimizer.calculate_wfe()
        aggregate_results = optimizer.get_aggregate_results()
        
        # Save results
        output_path = Path(f"results/{job_id}")
        output_path.mkdir(parents=True, exist_ok=True)
        optimizer.save_results(output_path / "wf_results.pkl")
        
        optimization_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Optimization completed",
            "results": {
                "wfe": wfe,
                "aggregate": aggregate_results,
                "num_periods": len(periods)
            }
        }
        
        logger.info(f"Optimization {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Optimization {job_id} failed: {e}", exc_info=True)
        optimization_jobs[job_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Optimization failed: {str(e)}"
        }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of optimization job."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return optimization_jobs[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get results of completed optimization."""
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = optimization_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job status is {job['status']}, not completed"
        )
    
    return job.get("results", {})


@app.get("/api/jobs")
async def list_jobs():
    """List all optimization jobs."""
    return {
        "jobs": [
            {"job_id": job_id, **status}
            for job_id, status in optimization_jobs.items()
        ]
    }


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload data file for optimization."""
    try:
        # Save uploaded file
        upload_path = Path("data/raw") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "filename": file.filename,
            "path": str(upload_path),
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """Download optimization results."""
    results_path = Path(f"results/{job_id}/wf_results.pkl")
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        path=results_path,
        filename=f"{job_id}_results.pkl",
        media_type="application/octet-stream"
    )


# Run server
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
