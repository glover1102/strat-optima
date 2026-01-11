# QTAlgo Super26 Quick Reference Guide

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Create a CSV file with OHLCV data:
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,100.0,102.0,99.0,101.0,1000000
...
```

### 3. Run Backtest
```bash
python main.py backtest --data data/raw/BTCUSD.csv --symbol BTCUSD
```

### 4. Run Walk-Forward Optimization
```bash
python main.py optimize --data data/raw/BTCUSD.csv --symbol BTCUSD
```

## Key Parameters

### Entry Parameters
- `strongTrendMinScore` (1.5): Lower threshold for strong trends
- `weakTrendMinScore` (3.0): Higher threshold for weak trends

### Risk Management
- `stopLossPercent` (2.0): Initial stop loss
- `takeProfitPercent` (4.0): Final target
- `partialExitPercent` (1.0): First profit target
- `trailingStopPercent` (0.8): Trailing stop distance

### Indicator Weights
All default to 1.0:
- `w_adx`: ADX indicator weight
- `w_regime`: Regime filter weight
- `w_pivotTrend`: Pivot trend weight
- `w_trendDuration`: Trend duration weight
- `w_mlSupertrend`: ML SuperTrend weight
- `w_linReg`: Linear regression weight
- `w_pivotLevels`: Pivot levels weight

## Configuration Files

### strategy_params.yaml
Contains all strategy parameters and optimization ranges.

### optimization_config.yaml
Contains walk-forward settings:
- `mode`: "rolling" or "anchored"
- `train_period_months`: 12
- `test_period_months`: 3
- `algorithm`: "optuna" or "random_search"
- `n_trials`: 100

## Code Examples

### Load and Prepare Data
```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_csv('data/raw/BTCUSD.csv', 'BTCUSD')
df = loader.clean_data(df)
```

### Calculate Indicators
```python
from src.strategy.indicators import calculate_all_indicators

params = {...}  # Your parameters
df_with_indicators = calculate_all_indicators(df, params)
```

### Generate Signals
```python
from src.strategy.signals import generate_signals

df_with_signals = generate_signals(df_with_indicators, params)
```

### Simulate Trades
```python
from src.strategy.exits import simulate_exits

trades_df = simulate_exits(df, params, df_with_signals)
```

### Calculate Metrics
```python
from src.optimization.metrics import calculate_all_metrics

metrics = calculate_all_metrics(trades_df)
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### Run Walk-Forward Optimization
```python
from src.optimization.walk_forward import WalkForwardOptimizer
from src.optimization.parameter_space import ParameterSpace

# Setup
param_space = ParameterSpace('config/strategy_params.yaml')
optimizer = WalkForwardOptimizer(config)

# Run
periods = optimizer.run_walk_forward(df, param_space, strategy_function)

# Results
wfe = optimizer.calculate_wfe()
results = optimizer.get_aggregate_results()
```

## Performance Metrics Explained

### Sharpe Ratio
Risk-adjusted return. Higher is better. 
- < 1.0: Poor
- 1.0-2.0: Good
- > 2.0: Excellent

### Maximum Drawdown
Largest peak-to-trough decline. Lower is better.
- < 10%: Excellent
- 10-20%: Good
- > 20%: High risk

### Win Rate
Percentage of winning trades.
- 40-45%: Acceptable with good risk/reward
- 50-60%: Good
- > 60%: Excellent

### Profit Factor
Gross profit / Gross loss.
- < 1.0: Losing strategy
- 1.5-2.0: Good
- > 2.0: Excellent

### Walk-Forward Efficiency (WFE)
OOS return / IS return ratio.
- > 1.0: Excellent (OOS better than IS)
- 0.5-1.0: Good
- < 0.5: Poor (possible overfitting)

## Directory Structure

```
results/
├── BTCUSD_report.txt          # Text performance report
├── BTCUSD_trades.csv          # Trade details
├── BTCUSD_wf_report.txt       # Walk-forward report
├── BTCUSD_param_stability.csv # Parameter evolution
├── BTCUSD_wf_results.pkl      # Saved optimizer state
├── BTCUSD_wf_analysis.html    # Interactive charts
└── BTCUSD_param_evolution.html
```

## Common Issues

### "No module named 'numpy'"
Install dependencies: `pip install -r requirements.txt`

### "Insufficient data for period"
Increase data history or reduce walk-forward periods

### "All trials failed"
Check parameter bounds and ensure data quality

### Low WFE
- Reduce number of optimized parameters
- Increase out-of-sample period
- Check for data quality issues

## Tips for Best Results

1. **Data Quality**: Use clean, high-quality data
2. **Sample Size**: 5+ years for robust optimization
3. **Parameter Count**: Optimize only critical parameters
4. **Walk-Forward Settings**: 
   - Training: 12 months minimum
   - Testing: 3 months
   - Step: 3 months for rolling
5. **Validation**: Always check WFE and parameter stability
6. **Multiple Instruments**: Test on different markets
7. **Out-of-Sample**: Never optimize on test period

## Performance Optimization

### Speed Up Optimization
- Reduce `n_trials`
- Use fewer walk-forward periods
- Optimize fewer parameters
- Use `n_jobs=-1` for parallel processing

### Memory Management
- Process data in chunks for very large datasets
- Save intermediate results
- Clear cache between runs

## Next Steps

1. Review `notebooks/strategy_analysis.ipynb` for examples
2. Customize parameters in `config/strategy_params.yaml`
3. Run backtests on your data
4. Analyze results and iterate
5. Deploy to Railway for automation

## Support

- Check README.md for detailed documentation
- Review test files for code examples
- Open GitHub issues for bugs
- Consult Jupyter notebook for workflow
