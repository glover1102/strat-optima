"""
Simple backtesting engine for the QTAlgo Super26 Strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

from .strategy.indicators import calculate_all_indicators
from .strategy.signals import generate_all_signals
from .strategy.exits import ExitManager, Position

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simple backtesting engine that runs the strategy on historical data.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for the backtest
        """
        self.initial_capital = initial_capital
        self.equity_curve = None
        self.trades = None
    
    def run(self, df: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run backtest with given data and parameters.
        
        Args:
            df: DataFrame with OHLCV data
            params: Strategy parameters
            
        Returns:
            Tuple of (equity_curve, trades_df)
        """
        logger.info(f"Starting backtest with {len(df)} bars")
        
        # Calculate indicators
        logger.info("Calculating indicators...")
        df_with_indicators = calculate_all_indicators(df.copy(), params)
        
        # Generate signals
        logger.info("Generating signals...")
        df_with_signals = generate_all_signals(df_with_indicators, params)
        
        # Run simulation
        logger.info("Running simulation...")
        equity_curve, trades_df = self._simulate_trading(df_with_signals, params)
        
        self.equity_curve = equity_curve
        self.trades = trades_df
        
        logger.info(f"Backtest complete. Total trades: {len(trades_df)}")
        
        return equity_curve, trades_df
    
    def _simulate_trading(self, df: pd.DataFrame, params: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Simulate trading based on signals.
        
        Args:
            df: DataFrame with signals
            params: Strategy parameters
            
        Returns:
            Tuple of (equity_curve, trades_df)
        """
        equity = [self.initial_capital]
        trades = []
        position = None
        exit_manager = ExitManager(params)
        
        for i in range(1, len(df)):
            current_equity = equity[-1]
            
            # Check for entry signal (only if not in position)
            if position is None and df['entry_signal'].iloc[i] != 0:
                position = Position(
                    entry_price=df['close'].iloc[i],
                    entry_bar=i,
                    direction=int(df['entry_signal'].iloc[i]),
                    size=1.0
                )
                # Calculate exit levels
                position = exit_manager.calculate_exit_levels(position)
                logger.debug(f"Entered {['SHORT', 'FLAT', 'LONG'][position.direction+1]} at {position.entry_price:.2f}")
            
            # Update position if we have one
            if position is not None:
                # Update trailing stop
                position = exit_manager.update_trailing_stop(position, df['close'].iloc[i])
                
                # Check for exit
                should_exit, exit_reason, exit_price = exit_manager.check_exit(
                    position, 
                    df.iloc[i]
                )
                
                if should_exit:
                    # Calculate PnL
                    if position.direction == 1:
                        pnl_pct = (exit_price - position.entry_price) / position.entry_price
                    else:
                        pnl_pct = (position.entry_price - exit_price) / position.entry_price
                    
                    pnl = current_equity * pnl_pct * position.size
                    current_equity += pnl
                    
                    trades.append({
                        'entry_date': df.index[position.entry_bar],
                        'exit_date': df.index[i],
                        'entry_price': position.entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG' if position.direction == 1 else 'SHORT',
                        'pnl': pnl,
                        'pnl_percent': pnl_pct * 100,
                        'exit_reason': exit_reason,
                        'bars_held': i - position.entry_bar
                    })
                    
                    logger.debug(f"Exited at {exit_price:.2f}, PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%), Reason: {exit_reason}")
                    
                    position = None
            
            equity.append(current_equity)
        
        # Close any remaining position at the last bar
        if position is not None:
            exit_price = df['close'].iloc[-1]
            if position.direction == 1:
                pnl_pct = (exit_price - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - exit_price) / position.entry_price
            
            pnl = equity[-1] * pnl_pct * position.size
            equity[-1] += pnl
            
            trades.append({
                'entry_date': df.index[position.entry_bar],
                'exit_date': df.index[-1],
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'direction': 'LONG' if position.direction == 1 else 'SHORT',
                'pnl': pnl,
                'pnl_percent': pnl_pct * 100,
                'exit_reason': 'End of data',
                'bars_held': len(df) - 1 - position.entry_bar
            })
        
        # Convert to pandas objects
        equity_curve = pd.Series(equity, index=df.index[:len(equity)])
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return equity_curve, trades_df
