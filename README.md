# QTAlgo Super26 Strategy Backtesting Environment

## Overview

This is a simple Python-based backtesting environment for the **QTAlgo Super26 Strategy**. It provides a streamlined way to test the strategy against historical OHLCV (Open, High, Low, Close, Volume) data and evaluate its performance.

The strategy combines 7 technical indicators with a dynamic scoring system to generate entry signals and uses a 3-stage exit management system for trade management.

## Quick Start

Get up and running in 3 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run backtest with sample data
python main.py --data data/sample_data.csv

# 3. View results
# - Performance metrics printed to console
# - Trade details in backtest_trades.csv
# - Equity curve in backtest_equity.csv
```

## Features

- **Complete Strategy Implementation**: All 7 indicators (ADX, Regime Filter, Pivot Trend, Trend Duration, ML SuperTrend, Linear Regression, Pivot Levels)
- **Dynamic Signal Generation**: Multi-indicator scoring system with ADX-based penalties
- **3-Stage Exit Management**: Initial stop loss, partial profit taking, trailing stops
- **CSV Data Loading**: Load 5+ years of historical OHLCV data from CSV files
- **Simple Configuration**: YAML-based strategy parameter configuration
- **Performance Metrics**: Key metrics including Total Return, Max Drawdown, Win Rate, and Profit Factor

## Project Structure

```
strat-optima/
├── src/
│   ├── strategy/
│   │   ├── indicators.py      # All 7 technical indicators
│   │   ├── signals.py         # Signal generation logic
│   │   └── exits.py           # 3-stage exit management
│   ├── data/
│   │   └── loader.py          # CSV data handling
│   ├── backtest.py            # Backtesting engine
│   └── metrics.py             # Performance calculations
├── config/
│   └── strategy_params.yaml   # Strategy parameters
├── data/                      # Place your CSV data files here
├── requirements.txt           # Python dependencies
├── main.py                    # Main entry point
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/glover1102/strat-optima.git
cd strat-optima

# Install Python dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data
```

## Data Format

The backtester expects CSV files with the following columns:

- **date**: Date/timestamp column (will be used as index)
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price  
- **close**: Closing price
- **volume**: Trading volume (optional, will be set to 0 if missing)

Example CSV format:
```csv
date,open,high,low,close,volume
2019-01-02,154.89,158.85,154.23,157.92,37039700
2019-01-03,143.98,145.72,142.00,142.19,91312200
...
```

Place your CSV data files in the `data/` directory.

## Usage

### Basic Backtest

Run a backtest with default parameters:

```bash
python main.py --data data/your_data.csv
```

### Custom Configuration

Specify a custom configuration file:

```bash
python main.py --data data/your_data.csv --config config/strategy_params.yaml
```

### Custom Initial Capital

Set a different initial capital amount:

```bash
python main.py --data data/your_data.csv --initial-capital 50000
```

### Command-Line Options

- `--data`: Path to CSV file with OHLCV data (default: `data/ohlcv_data.csv`)
- `--config`: Path to strategy configuration file (default: `config/strategy_params.yaml`)
- `--initial-capital`: Initial capital for backtest (default: `100000`)

## Configuration

### Strategy Parameters (`config/strategy_params.yaml`)

The strategy can be customized by modifying the parameters in the configuration file:

**Entry Signal Parameters:**
- `strongTrendMinScore`: Minimum score for strong trend entries (default: 1.5)
- `weakTrendMinScore`: Minimum score for weak trend entries (default: 3.0)

**Exit Management Parameters:**
- `stopLossPercent`: Initial stop loss percentage (default: 2.0)
- `takeProfitPercent`: Final take profit percentage (default: 4.0)
- `partialExitPercent`: First take profit level (default: 1.0)
- `trailingStopPercent`: Trailing stop distance (default: 0.8)

**Indicator Weights:**
- `w_adx`: ADX weight (default: 1.0)
- `w_regime`: Regime filter weight (default: 1.0)
- `w_pivotTrend`: Pivot trend weight (default: 1.5)
- `w_trendDuration`: Trend duration weight (default: 0.8)
- `w_mlSupertrend`: ML SuperTrend weight (default: 1.2)
- `w_linregChannel`: Linear regression weight (default: 0.9)
- `w_pivotLevels`: Pivot levels weight (default: 0.7)

And many more indicator-specific parameters. See `config/strategy_params.yaml` for the complete list.

## Performance Metrics

The backtester outputs the following key metrics:

**Return Metrics:**
- Total Return: Overall percentage return
- Sharpe Ratio: Risk-adjusted return metric

**Risk Metrics:**
- Max Drawdown: Maximum peak-to-trough decline

**Trade Metrics:**
- Total Trades: Number of trades executed
- Win Rate: Percentage of profitable trades
- Profit Factor: Ratio of gross profit to gross loss
- Average Win/Loss: Average P&L for winning and losing trades

## Output Files

After running a backtest, the following files are generated:

- `backtest_trades.csv`: Detailed trade-by-trade results
- `backtest_equity.csv`: Equity curve over time

## Strategy Logic

The QTAlgo Super26 Strategy uses a multi-indicator approach:

1. **Indicator Calculation**: Seven technical indicators are calculated on the OHLCV data
2. **Signal Generation**: Each indicator contributes to a composite score based on its weight
3. **Entry Logic**: Trades are entered when the composite score exceeds minimum thresholds
4. **Exit Management**: Positions are managed with a 3-stage system:
   - Initial stop loss for risk management
   - Partial profit taking at first target
   - Trailing stop for remaining position

## Example

Here's a complete example workflow:

```bash
# 1. Prepare your data
# Place a CSV file with 5 years of OHLCV data in the data/ directory
# Example: data/SPY_5years.csv

# 2. Run the backtest
python main.py --data data/SPY_5years.csv --initial-capital 100000

# 3. Review the output
# - Performance metrics will be printed to console
# - Trade details saved to backtest_trades.csv
# - Equity curve saved to backtest_equity.csv
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

**Important**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading strategy with real capital. The authors are not responsible for any financial losses incurred from using this software.

## Support

- **Issues**: [GitHub Issues](https://github.com/glover1102/strat-optima/issues)
- **Discussions**: [GitHub Discussions](https://github.com/glover1102/strat-optima/discussions)

## Acknowledgments

**Built for quantitative traders and researchers**
