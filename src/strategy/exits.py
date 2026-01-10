"""
Exit Management System for QTAlgo Super26 Strategy

Implements 3-stage exit management with:
1. Initial stop loss
2. Partial profit taking
3. Trailing stop
4. Signal-based exits
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ExitReason(Enum):
    """Enumeration of exit reasons."""
    STOP_LOSS = "stop_loss"
    PARTIAL_PROFIT = "partial_profit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    SIGNAL_WEAKENING = "signal_weakening"
    END_OF_DATA = "end_of_data"


@dataclass
class Position:
    """Represents an open trading position."""
    entry_price: float
    entry_time: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    size: float
    initial_stop: float
    partial_target: float
    final_target: float
    trailing_stop: Optional[float] = None
    partial_closed: bool = False
    highest_price: Optional[float] = None  # For long positions
    lowest_price: Optional[float] = None   # For short positions


class ExitManager:
    """
    Manages position exits for the QTAlgo Super26 strategy.
    
    Implements a 3-stage exit system:
    1. Stop Loss: Initial risk management
    2. Partial Exit: Take partial profits at first target
    3. Trailing Stop: Trail remaining position to final target
    """
    
    def __init__(self, params: Dict):
        """
        Initialize exit manager with parameters.
        
        Args:
            params: Dictionary containing:
                - stopLossPercent: Stop loss percentage
                - partialExitPercent: Partial profit percentage
                - takeProfitPercent: Final take profit percentage
                - trailingStopPercent: Trailing stop distance
        """
        self.stop_loss_pct = params.get('stopLossPercent', 2.0) / 100
        self.partial_exit_pct = params.get('partialExitPercent', 1.0) / 100
        self.take_profit_pct = params.get('takeProfitPercent', 4.0) / 100
        self.trailing_stop_pct = params.get('trailingStopPercent', 0.8) / 100
        
        self.current_position: Optional[Position] = None
    
    def open_position(self, entry_price: float, entry_time: pd.Timestamp,
                     direction: int, size: float = 1.0) -> Position:
        """
        Open a new position with calculated exit levels.
        
        Args:
            entry_price: Entry price
            entry_time: Entry timestamp
            direction: 1 for long, -1 for short
            size: Position size
            
        Returns:
            Position object with all exit levels set
        """
        if direction == 1:  # Long position
            initial_stop = entry_price * (1 - self.stop_loss_pct)
            partial_target = entry_price * (1 + self.partial_exit_pct)
            final_target = entry_price * (1 + self.take_profit_pct)
            highest_price = entry_price
            lowest_price = None
        else:  # Short position
            initial_stop = entry_price * (1 + self.stop_loss_pct)
            partial_target = entry_price * (1 - self.partial_exit_pct)
            final_target = entry_price * (1 - self.take_profit_pct)
            highest_price = None
            lowest_price = entry_price
        
        position = Position(
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            size=size,
            initial_stop=initial_stop,
            partial_target=partial_target,
            final_target=final_target,
            highest_price=highest_price,
            lowest_price=lowest_price
        )
        
        self.current_position = position
        return position
    
    def update_trailing_stop(self, position: Position, 
                            current_price: float) -> Position:
        """
        Update trailing stop based on current price.
        
        Only activates after partial profit is taken.
        
        Args:
            position: Current position
            current_price: Current market price
            
        Returns:
            Updated position
        """
        if not position.partial_closed:
            return position
        
        if position.direction == 1:  # Long position
            # Update highest price
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
            
            # Calculate trailing stop
            new_trailing = position.highest_price * (1 - self.trailing_stop_pct)
            
            # Update if higher than current trailing stop
            if position.trailing_stop is None or new_trailing > position.trailing_stop:
                position.trailing_stop = new_trailing
        
        else:  # Short position
            # Update lowest price
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Calculate trailing stop
            new_trailing = position.lowest_price * (1 + self.trailing_stop_pct)
            
            # Update if lower than current trailing stop
            if position.trailing_stop is None or new_trailing < position.trailing_stop:
                position.trailing_stop = new_trailing
        
        return position
    
    def check_exit(self, position: Position, current_price: float,
                   signal_reversal: bool = False, 
                   signal_weakening: bool = False) -> Tuple[bool, Optional[ExitReason], float]:
        """
        Check if position should be exited.
        
        Checks in priority order:
        1. Stop loss
        2. Signal reversal
        3. Take profit (full or partial)
        4. Trailing stop
        5. Signal weakening
        
        Args:
            position: Current position
            current_price: Current market price
            signal_reversal: Whether signal has reversed
            signal_weakening: Whether signal is weakening
            
        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        if position.direction == 1:  # Long position
            # Check stop loss
            if current_price <= position.initial_stop:
                return True, ExitReason.STOP_LOSS, position.initial_stop
            
            # Check signal reversal (immediate exit)
            if signal_reversal:
                return True, ExitReason.SIGNAL_REVERSAL, current_price
            
            # Check take profit
            if current_price >= position.final_target:
                return True, ExitReason.TAKE_PROFIT, position.final_target
            
            # Check partial profit
            if not position.partial_closed and current_price >= position.partial_target:
                return True, ExitReason.PARTIAL_PROFIT, position.partial_target
            
            # Check trailing stop (only after partial)
            if position.partial_closed and position.trailing_stop is not None:
                if current_price <= position.trailing_stop:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop
            
            # Check signal weakening (only after partial)
            if position.partial_closed and signal_weakening:
                return True, ExitReason.SIGNAL_WEAKENING, current_price
        
        else:  # Short position
            # Check stop loss
            if current_price >= position.initial_stop:
                return True, ExitReason.STOP_LOSS, position.initial_stop
            
            # Check signal reversal (immediate exit)
            if signal_reversal:
                return True, ExitReason.SIGNAL_REVERSAL, current_price
            
            # Check take profit
            if current_price <= position.final_target:
                return True, ExitReason.TAKE_PROFIT, position.final_target
            
            # Check partial profit
            if not position.partial_closed and current_price <= position.partial_target:
                return True, ExitReason.PARTIAL_PROFIT, position.partial_target
            
            # Check trailing stop (only after partial)
            if position.partial_closed and position.trailing_stop is not None:
                if current_price >= position.trailing_stop:
                    return True, ExitReason.TRAILING_STOP, position.trailing_stop
            
            # Check signal weakening (only after partial)
            if position.partial_closed and signal_weakening:
                return True, ExitReason.SIGNAL_WEAKENING, current_price
        
        return False, None, current_price
    
    def execute_partial_exit(self, position: Position) -> Position:
        """
        Execute partial exit, closing half the position.
        
        Args:
            position: Current position
            
        Returns:
            Updated position with reduced size
        """
        position.size = position.size * 0.5
        position.partial_closed = True
        return position
    
    def close_position(self) -> None:
        """Close the current position."""
        self.current_position = None
    
    def has_position(self) -> bool:
        """Check if there is an open position."""
        return self.current_position is not None


def simulate_exits(df: pd.DataFrame, params: Dict, 
                   signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate exit management on historical data.
    
    Args:
        df: DataFrame with OHLC data
        params: Strategy parameters
        signals_df: DataFrame with entry signals and signal analysis
        
    Returns:
        DataFrame with exit information for each trade
    """
    exit_manager = ExitManager(params)
    trades = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        signal_row = signals_df.iloc[idx]
        
        # Check if we have a position
        if exit_manager.has_position():
            position = exit_manager.current_position
            
            # Update trailing stop
            position = exit_manager.update_trailing_stop(position, row['close'])
            
            # Check for exit
            should_exit, exit_reason, exit_price = exit_manager.check_exit(
                position,
                row['close'],
                signal_reversal=signal_row.get('signal_reversal', False),
                signal_weakening=signal_row.get('signal_weakening', False)
            )
            
            if should_exit:
                # Record trade
                pnl = (exit_price - position.entry_price) * position.direction * position.size
                pnl_pct = pnl / position.entry_price * 100
                
                trades.append({
                    'entry_time': position.entry_time,
                    'entry_price': position.entry_price,
                    'exit_time': row.name,
                    'exit_price': exit_price,
                    'direction': position.direction,
                    'size': position.size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason.value,
                    'duration': (row.name - position.entry_time).total_seconds() / 3600,  # hours
                })
                
                # Handle partial exit
                if exit_reason == ExitReason.PARTIAL_PROFIT:
                    exit_manager.execute_partial_exit(position)
                else:
                    exit_manager.close_position()
        
        # Check for new entry signal
        if not exit_manager.has_position() and signal_row['signal'] != 0:
            entry_price = row['close']
            exit_manager.open_position(
                entry_price,
                row.name,
                int(signal_row['signal']),
                size=1.0
            )
    
    # Close any remaining position at end
    if exit_manager.has_position():
        position = exit_manager.current_position
        last_row = df.iloc[-1]
        exit_price = last_row['close']
        pnl = (exit_price - position.entry_price) * position.direction * position.size
        pnl_pct = pnl / position.entry_price * 100
        
        trades.append({
            'entry_time': position.entry_time,
            'entry_price': position.entry_price,
            'exit_time': last_row.name,
            'exit_price': exit_price,
            'direction': position.direction,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': ExitReason.END_OF_DATA.value,
            'duration': (last_row.name - position.entry_time).total_seconds() / 3600,
        })
    
    return pd.DataFrame(trades)
