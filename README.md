# QTAlgo Super26 Strategy Walk-Forward Optimization Framework

## Overview

The **QTAlgo Super26 Strategy Walk-Forward Optimization Framework** is a comprehensive Python-based walk-forward optimization system for the QTAlgo Super26 trading strategy. This framework provides advanced backtesting capabilities with rigorous out-of-sample validation to ensure strategy performance reliability and minimize overfitting risks.

## Key Features

- **Complete Strategy Implementation**: All 7 indicators from QTAlgo Super26 (ADX, Regime Filter, Pivot Trend, Trend Duration, ML SuperTrend, Linear Regression, Pivot Levels)
- **Dynamic Signal Generation**: Sophisticated scoring system with ADX-based penalty for weak trends
- **3-Stage Exit Management**: Stop loss, partial profit taking, and trailing stops
- **Walk-Forward Optimization**: Both rolling and anchored window approaches
- **Multi-Objective Optimization**: Optuna-based optimization with configurable objectives
- **Comprehensive Analytics**: Detailed performance metrics and visualization tools
- **Production-Ready**: Docker support and Railway deployment configuration

## Project Structure

```
strat-optima/
├── src/
│   ├── strategy/
│   │   ├── indicators.py       # All 7 technical indicators
│   │   ├── signals.py          # Signal generation with dynamic scoring
│   │   └── exits.py            # 3-stage exit management system
│   ├── optimization/
│   │   ├── walk_forward.py     # Walk-forward optimization engine
│   │   ├── parameter_space.py  # Parameter definitions and sampling
│   │   └── metrics.py          # Performance calculations
│   ├── data/
│   │   └── loader.py           # OHLCV data handling
│   └── utils/
│       ├── plotting.py         # Visualization utilities
│       └── reporting.py        # Report generation
├── config/
│   ├── strategy_params.yaml    # Strategy configuration
│   └── optimization_config.yaml # Optimization settings
├── tests/
│   └── test_strategy.py        # Unit tests
├── notebooks/
│   └── strategy_analysis.ipynb # Analysis notebook
├── main.py                     # CLI entry point
├── requirements.txt
├── Dockerfile
└── railway.json
```

## Installation

### Standard Installation

```bash
git clone https://github.com/glover1102/strat-optima.git
cd strat-optima
pip install -r requirements.txt
```

### Docker Installation

```bash
docker build -t strat-optima .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results strat-optima
```

## Usage

### Command Line Interface

#### Single Backtest
Run a backtest with default parameters:

```bash
python main.py backtest --data data/raw/BTCUSD.csv --symbol BTCUSD --output results/
```

#### Walk-Forward Optimization
Run walk-forward optimization:

```bash
python main.py optimize \
    --strategy-config config/strategy_params.yaml \
    --opt-config config/optimization_config.yaml \
    --data data/raw/BTCUSD.csv \
    --symbol BTCUSD \
    --output results/
```

### Python API

```python
from src.strategy.indicators import calculate_all_indicators
from src.strategy.signals import generate_signals
from src.strategy.exits import simulate_exits
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import ParameterSpace
from src.data.loader import DataLoader

# Load data
loader = DataLoader()
df = loader.load_csv('data/raw/BTCUSD.csv', 'BTCUSD')

# Setup parameters
param_space = ParameterSpace('config/strategy_params.yaml')

# Create strategy function
def strategy_func(data, params):
    df_ind = calculate_all_indicators(data, params)
    df_sig = generate_signals(df_ind, params)
    trades = simulate_exits(data, params, df_sig)
    return trades

# Run optimization
optimizer = WalkForwardOptimizer(config)
periods = optimizer.run_walk_forward(df, param_space, strategy_func)

# Get results
wfe = optimizer.calculate_wfe()
results = optimizer.get_aggregate_results()
```

### Jupyter Notebook

See `notebooks/strategy_analysis.ipynb` for a complete interactive analysis example.

## Strategy Components

### 1. Indicators (7 Components)

- **ADX (Trend Strength Filter)**: Measures trend strength with directional movement
- **Regime Filter**: HMA-based trend and volume analysis
- **Pivot Trend**: Primary signal generator using pivot highs/lows with ATR offset
- **Trend Duration Forecast**: HMA-based trend persistence estimation
- **ML Adaptive SuperTrend**: Volatility-adaptive SuperTrend indicator
- **Linear Regression Channel**: Slope-based trend analysis with deviation bands
- **Pivot Levels**: Support/resistance confirmation with ATR proximity

### 2. Signal Generation

- **Dynamic Scoring System**: Combines all 7 indicators with configurable weights
- **ADX-Based Penalty**: Requires higher confirmation in weak trends
- **Multiple Entry Modes**: Strong trend (lower threshold) and weak trend (higher threshold) entries
- **Signal Validation**: Ensures sufficient data and valid indicator values

### 3. Exit Management (3 Stages)

- **Stage 1 - Stop Loss**: Initial risk management at entry
- **Stage 2 - Partial Profit**: Take 50% profit at first target
- **Stage 3 - Trailing Stop**: Trail remaining position to maximize gains
- **Signal-Based Exits**: Exit on signal reversal or significant weakening

## Configuration

### Strategy Parameters (`config/strategy_params.yaml`)

Key parameters to optimize:
- `strongTrendMinScore`: Minimum score for strong trend entries (default: 1.5)
- `weakTrendMinScore`: Minimum score for weak trend entries (default: 3.0)
- `stopLossPercent`: Initial stop loss percentage (default: 2.0)
- `takeProfitPercent`: Final take profit percentage (default: 4.0)
- `partialExitPercent`: First take profit level (default: 1.0)
- `trailingStopPercent`: Trailing stop distance (default: 0.8)
- Indicator weights (`w_adx`, `w_regime`, etc.)

### Optimization Settings (`config/optimization_config.yaml`)

- **Walk-Forward Mode**: `rolling` or `anchored`
- **Training Period**: 12 months (default)
- **Testing Period**: 3 months (default)
- **Optimization Algorithm**: `optuna` or `random_search`
- **Trials**: 100 (default)
- **Multi-Objective Weights**: Sharpe (40%), Drawdown (30%), Win Rate (20%), Profit Factor (10%)

## Performance Metrics

The framework calculates comprehensive metrics:

### Returns
- Total Return
- Annual Return
- Cumulative Return

### Risk Metrics
- Volatility
- Maximum Drawdown
- Average Drawdown
- Drawdown Duration

### Risk-Adjusted Returns
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Recovery Factor

### Trade Statistics
- Total Trades
- Win Rate
- Profit Factor
- Expectancy
- Average Win/Loss

### Walk-Forward Specific
- Walk-Forward Efficiency (WFE)
- Parameter Stability (coefficient of variation)
- In-Sample vs Out-of-Sample comparison

## Testing

Run the test suite:

```bash
pytest tests/test_strategy.py -v
```

Run with coverage:

```bash
pytest tests/test_strategy.py --cov=src --cov-report=html
```

## Railway Deployment

The framework is configured for Railway deployment:

1. Push to GitHub
2. Connect to Railway
3. Railway will automatically detect `railway.json` and deploy
4. Set environment variables as needed
5. Schedule automated optimization runs

## Data Requirements

### CSV Format
CSV files should have the following columns:
- `timestamp` or `date`: DateTime index
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

### Supported Data Sources
- CSV files (local)
- Yahoo Finance (via yfinance)
- Crypto exchanges (via ccxt)
- Custom data loaders

### Recommended Data
- Minimum: 2 years of historical data
- Optimal: 5+ years for robust walk-forward analysis
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d

## Visualization

The framework generates:
- Equity curves with benchmark comparison
- Drawdown analysis charts
- Walk-forward efficiency plots
- Parameter stability analysis
- Trade distribution histograms
- Signal analysis visualizations

All plots are generated using Plotly for interactive exploration.

## Performance Considerations

- **Vectorized Operations**: All indicators use pandas/numpy vectorization
- **Parallel Processing**: Walk-forward periods can be optimized in parallel
- **Memory Management**: Efficient data handling for large datasets
- **Caching**: Results cached to avoid redundant calculations

## Contributing

Contributions are welcome! Areas for improvement:
- Additional indicators
- Alternative optimization algorithms
- Enhanced visualization tools
- Performance optimizations
- Documentation improvements

## License

This project is licensed under the MIT License.

## Disclaimer

**IMPORTANT**: This framework is designed for educational and research purposes only. 

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Always conduct thorough testing before deploying with real capital
- This is not financial advice
- Use at your own risk

## Support

For questions or issues:
- Create an issue in this repository
- Review the example notebook in `notebooks/`
- Check the test files for usage examples

## Acknowledgments

Based on the QTAlgo Super26 strategy framework with comprehensive walk-forward optimization capabilities.