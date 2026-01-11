# Data Directory

Place your CSV files with OHLCV data in this directory.

## Expected Format

CSV files should contain the following columns:
- **date**: Date/timestamp (any format parseable by pandas)
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price
- **volume**: Trading volume (optional)

## Example

```csv
date,open,high,low,close,volume
2019-01-02,154.89,158.85,154.23,157.92,37039700
2019-01-03,143.98,145.72,142.00,142.19,91312200
2019-01-04,144.53,148.55,143.80,148.26,58607070
```

## Sample Data

A sample data file (`sample_data.csv`) is included for testing purposes. You can use it to verify the backtester is working:

```bash
python main.py --data data/sample_data.csv
```

## Getting Real Data

You can obtain historical OHLCV data from various sources:
- Yahoo Finance (download as CSV)
- Your broker's platform
- Data providers (Alpha Vantage, Quandl, etc.)
- Cryptocurrency exchanges (Binance, Coinbase, etc.)

Make sure your data:
1. Covers at least 1 year (preferably 5+ years)
2. Has consistent daily or intraday frequency
3. Contains valid OHLC relationships (high >= open/close, low <= open/close)
