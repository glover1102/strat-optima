"""
Data Loader for OHLCV Data

Handles loading, validation, and preprocessing of market data from various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of OHLCV data.
    
    Supports multiple data sources:
    - CSV files
    - Database connections
    - API feeds (yfinance, ccxt)
    """
    
    def __init__(self, data_path: Optional[str] = None,
                 timeframe: str = '1h'):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        """
        self.data_path = Path(data_path) if data_path else None
        self.timeframe = timeframe
    
    def load_csv(self, filepath: Union[str, Path], 
                 symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Expected CSV format:
        - timestamp, open, high, low, close, volume
        or
        - date, open, high, low, close, volume
        
        Args:
            filepath: Path to CSV file
            symbol: Optional symbol name to add to DataFrame
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Handle timestamp/date column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            # Assume first column is timestamp
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add symbol if provided
        if symbol:
            df['symbol'] = symbol
        
        # Sort by index
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def load_multiple_csv(self, symbols: List[str]) -> dict:
        """
        Load data for multiple symbols from CSV files.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if self.data_path is None:
            raise ValueError("data_path must be set to load multiple files")
        
        data_dict = {}
        
        for symbol in symbols:
            # Try common filename patterns
            patterns = [
                f"{symbol}.csv",
                f"{symbol}_{self.timeframe}.csv",
                f"{symbol.lower()}.csv",
                f"{symbol.upper()}.csv",
            ]
            
            filepath = None
            for pattern in patterns:
                test_path = self.data_path / pattern
                if test_path.exists():
                    filepath = test_path
                    break
            
            if filepath is None:
                logger.warning(f"Could not find data file for {symbol}")
                continue
            
            try:
                df = self.load_csv(filepath, symbol)
                data_dict[symbol] = df
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        
        return data_dict
    
    def load_from_yfinance(self, symbol: str, start_date: str,
                          end_date: Optional[str] = None,
                          interval: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1m, 5m, 15m, 1h, 1d, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")
        
        if interval is None:
            interval = self._convert_timeframe_to_yfinance(self.timeframe)
        
        logger.info(f"Downloading {symbol} from {start_date} to {end_date or 'present'}")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        df.rename(columns={'adj close': 'adj_close'}, inplace=True)
        
        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df['symbol'] = symbol
        
        logger.info(f"Downloaded {len(df)} rows")
        
        return df
    
    def load_from_ccxt(self, exchange: str, symbol: str,
                       start_date: str, end_date: Optional[str] = None,
                       timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from crypto exchange via CCXT.
        
        Args:
            exchange: Exchange name (e.g., 'binance', 'coinbase')
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt not installed. Install with: pip install ccxt")
        
        if timeframe is None:
            timeframe = self.timeframe
        
        logger.info(f"Fetching {symbol} from {exchange}")
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange)
        exchange_obj = exchange_class()
        
        # Convert dates to timestamps
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000) if end_date else None
        
        # Fetch OHLCV data
        all_data = []
        while True:
            data = exchange_obj.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not data:
                break
            
            all_data.extend(data)
            since = data[-1][0] + 1
            
            if until and since >= until:
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        
        logger.info(f"Fetched {len(df)} rows")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data quality.
        
        Checks for:
        - Missing values
        - Invalid OHLC relationships
        - Duplicate timestamps
        - Negative values
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for missing values
        null_counts = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
        if null_counts.any():
            issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for negative values
        negative_mask = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1)
        if negative_mask.any():
            issues.append(f"Found {negative_mask.sum()} rows with negative prices")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_ohlc.any():
            issues.append(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            issues.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")
        
        # Check for large gaps
        time_diff = df.index.to_series().diff()
        median_diff = time_diff.median()
        large_gaps = time_diff > (median_diff * 10)
        if large_gaps.any():
            issues.append(f"Found {large_gaps.sum()} large time gaps in data")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def clean_data(self, df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   fill_missing: bool = True) -> pd.DataFrame:
        """
        Clean and preprocess data.
        
        Args:
            df: DataFrame to clean
            drop_duplicates: Whether to drop duplicate timestamps
            fill_missing: Whether to fill missing values
            
        Returns:
            Cleaned DataFrame
        """
        result = df.copy()
        
        # Drop duplicates
        if drop_duplicates:
            result = result[~result.index.duplicated(keep='first')]
        
        # Fill missing values
        if fill_missing:
            # Forward fill first, then backward fill
            result = result.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN rows
        result = result.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure correct data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[col] = pd.to_numeric(result[col], errors='coerce')
        
        return result
    
    def resample_data(self, df: pd.DataFrame, 
                     target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: DataFrame to resample
            target_timeframe: Target timeframe (1h, 4h, 1d, etc.)
            
        Returns:
            Resampled DataFrame
        """
        # Convert timeframe to pandas offset
        offset = self._timeframe_to_offset(target_timeframe)
        
        # Resample
        resampled = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        return resampled
    
    @staticmethod
    def _convert_timeframe_to_yfinance(timeframe: str) -> str:
        """Convert internal timeframe to yfinance interval."""
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }
        return mapping.get(timeframe, '1h')
    
    @staticmethod
    def _timeframe_to_offset(timeframe: str) -> str:
        """Convert timeframe to pandas offset string."""
        mapping = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
        }
        return mapping.get(timeframe, '1H')


def prepare_data_for_backtest(df: pd.DataFrame, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare data for backtesting by filtering date range.
    
    Args:
        df: DataFrame with OHLCV data
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if start_date:
        result = result[result.index >= start_date]
    
    if end_date:
        result = result[result.index <= end_date]
    
    return result
