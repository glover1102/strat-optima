# QTAlgo Super26 Strategy Walk-Forward Optimization Framework

## Overview

The **QTAlgo Super26 Strategy Walk-Forward Optimization Framework** is a comprehensive Python-based algorithmic trading optimization system that implements robust walk-forward analysis for the QTAlgo Super26 trading strategy. This production-ready framework provides advanced backtesting capabilities with rigorous out-of-sample validation to ensure strategy performance reliability and minimize overfitting risks.

## Key Features

- **Complete Strategy Implementation**: All 7 indicators (ADX, Regime Filter, Pivot Trend, Trend Duration, ML SuperTrend, Linear Regression, Pivot Levels)
- **Dynamic Signal Generation**: Multi-indicator scoring system with ADX-based penalties
- **3-Stage Exit Management**: Initial stop loss, partial profit taking, trailing stops
- **Walk-Forward Optimization**: Both anchored and rolling window approaches
- **Parameter Stability Tracking**: Monitor parameter consistency across windows
- **Comprehensive Metrics**: Sharpe ratio, max drawdown, win rate, profit factor, and more
- **Interactive Visualizations**: Plotly-based dashboards and analysis tools
- **Production Deployment**: Docker and Railway.app ready with FastAPI web interface

## Project Structure

```
strat-optima/
├── src/
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── indicators.py      # All 7 technical indicators
│   │   ├── signals.py         # Signal generation logic
│   │   └── exits.py           # 3-stage exit management
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── walk_forward.py    # Main WFO engine
│   │   ├── parameter_space.py # Parameter definitions
│   │   └── metrics.py         # Performance calculations
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # OHLCV data handling
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py        # Visualization tools
│       └── reporting.py       # Results analysis
├── config/
│   ├── strategy_params.yaml       # Strategy parameters
│   └── optimization_config.yaml   # Optimization settings
├── tests/
│   └── test_strategy.py          # Test suite
├── notebooks/
│   └── strategy_analysis.ipynb   # Analysis notebooks
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
├── railway.json                  # Railway deployment config
├── main.py                       # Main entry point
└── README.md
```

## Installation

### Prerequisites
- Python 3.11+
- TA-Lib (for technical indicators)
- Sufficient computational resources for optimization

### Local Installation

```bash
# Clone the repository
git clone https://github.com/glover1102/strat-optima.git
cd strat-optima

# Install Python dependencies
pip install -r requirements.txt

# For TA-Lib on macOS
brew install ta-lib

# For TA-Lib on Ubuntu/Debian
sudo apt-get install ta-lib

# Create necessary directories
mkdir -p logs results data
```

### Docker Installation

```bash
# Build Docker image
docker build -t strat-optima .

# Run container
docker run -p 8000:8000 -v $(pwd)/results:/app/results strat-optima
```

## Usage

### Command-Line Interface

**Run optimization for single symbol:**
```bash
python main.py --symbols SPY --start-date 2018-01-01 --n-trials 100
```

**Run optimization for multiple symbols:**
```bash
python main.py --symbols SPY QQQ IWM --start-date 2018-01-01 --n-trials 50
```

**Use different optimization algorithm:**
```bash
python main.py --symbols SPY --algorithm optuna --n-trials 100
```

### Web Interface (FastAPI)

**Start the server:**
```bash
python main.py --mode server --port 8000
```

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `GET /status` - Get optimization status
- `POST /optimize` - Start new optimization
- `GET /results` - Get optimization results

**Example API request:**
```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["SPY"],
    "start_date": "2018-01-01",
    "n_trials": 100,
    "algorithm": "optuna"
  }'
```

### Python API

```python
from src.data.loader import DataLoader
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import create_parameter_space_from_config
from src.utils.reporting import print_summary_report
from src.utils.plotting import create_dashboard

# Load data
loader = DataLoader(source='yfinance')
df = loader.load_data('SPY', start_date='2018-01-01', end_date='2023-12-31')

# Create parameter space
param_space = create_parameter_space_from_config('config/optimization_config.yaml')

# Initialize optimizer
optimizer = WalkForwardOptimizer(
    window_type='rolling',
    train_period_months=12,
    test_period_months=3,
    step_size_months=3
)

# Run optimization
results = optimizer.run_walk_forward(
    df,
    param_space,
    strategy_func=run_backtest,
    objective_func=objective_function,
    n_trials=100,
    algorithm='optuna'
)

# Display results
print_summary_report(results)

# Create visualizations
create_dashboard(results, combined_equity, param_history, 'results/dashboard')
```

## Configuration

### Strategy Parameters (`config/strategy_params.yaml`)

Key parameters to optimize:
- `strongTrendMinScore`: Minimum score for strong trend entries (default: 1.5)
- `weakTrendMinScore`: Minimum score for weak trend entries (default: 3.0)
- `stopLossPercent`: Initial stop loss percentage (default: 2.0)
- `takeProfitPercent`: Final take profit percentage (default: 4.0)
- `adxThreshold`: ADX trend strength threshold (default: 20)
- Indicator weights: `w_adx`, `w_regime`, `w_pivotTrend`, etc.

### Optimization Configuration (`config/optimization_config.yaml`)

Walk-forward settings:
- `window_type`: 'rolling' or 'anchored'
- `train_period_months`: Training period (default: 12)
- `test_period_months`: Testing period (default: 3)
- `step_size_months`: Step size for rolling window (default: 3)
- `algorithm`: 'optuna', 'grid_search', or 'random_search'
- `n_trials`: Number of optimization trials (default: 100)

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_strategy.py::TestIndicators -v
```

## Performance Metrics

The framework calculates comprehensive metrics:

**Return Metrics:**
- Total Return
- Annual Return
- Risk-adjusted returns

**Risk Metrics:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Ulcer Index

**Trade Metrics:**
- Win Rate
- Profit Factor
- Average Win/Loss
- Total Trades
- Expectancy

**Walk-Forward Metrics:**
- Walk-Forward Efficiency (WFE)
- Parameter Stability
- IS vs OOS degradation

## Visualization

The framework generates interactive visualizations:

1. **Equity Curve**: Combined out-of-sample equity with drawdown
2. **Parameter Evolution**: Parameter changes across windows
3. **Walk-Forward Efficiency**: IS vs OOS comparison
4. **Performance Metrics**: Bar charts of key metrics
5. **Signal Distribution**: Entry signal analysis
6. **Parameter Stability**: Stability scores

## Railway Deployment

Deploy to Railway.app:

1. Push code to GitHub
2. Connect Railway to your repository
3. Railway will automatically detect `railway.json` and `Dockerfile`
4. Set environment variables if needed
5. Deploy!

The API will be available at your Railway URL.

## Examples

See the `notebooks/` directory for detailed examples:
- `strategy_analysis.ipynb`: Strategy indicator and signal analysis
- `optimization_results.ipynb`: Walk-forward optimization results

## Roadmap

- [ ] Add more optimization algorithms (genetic, differential evolution)
- [ ] Implement position sizing strategies
- [ ] Add live trading integration
- [ ] Enhance multi-asset portfolio optimization
- [ ] Add machine learning enhancements
- [ ] Real-time monitoring dashboard

## Performance Tips

1. **Use Optuna** for efficient parameter search
2. **Parallel processing**: Set `n_jobs=-1` in config
3. **Reduce trials** for faster initial testing
4. **Cache data**: Load data once and reuse
5. **Use smaller windows** for rapid prototyping

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

**Important**: This framework is designed for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading strategy with real capital. The authors are not responsible for any financial losses incurred from using this software.

## Support

- **Issues**: [GitHub Issues](https://github.com/glover1102/strat-optima/issues)
- **Discussions**: [GitHub Discussions](https://github.com/glover1102/strat-optima/discussions)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{strat_optima,
  title={QTAlgo Super26 Walk-Forward Optimization Framework},
  author={QTAlgo Team},
  year={2024},
  url={https://github.com/glover1102/strat-optima}
}
```

---

**Built with ❤️ for quantitative traders and researchers**