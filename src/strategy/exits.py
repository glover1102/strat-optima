"""
QTAlgo Super26 Strategy - Exit Management Module

This module implements the 3-stage exit management system:
1. Initial stop loss
2. Partial profit taking at first target
3. Trailing stop with final profit target
4. Signal-based exit triggers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Position:
    """Represents an open trading position"""
    entry_price: float
    entry_bar: int
    direction: int  # 1 for long, -1 for short
    size: float = 1.0
    initial_stop: float = 0.0
    partial_exit_target: float = 0.0
    final_exit_target: float = 0.0
    trailing_stop: float = 0.0
    partial_exit_filled: bool = False
    remaining_size: float = 1.0
    highest_price: float = 0.0  # For long trailing
    lowest_price: float = float('inf')  # For short trailing
    

class ExitManager:
    """Manages exit logic for open positions"""
    
    def __init__(self, params: Dict):
        """
        Initialize exit manager
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.stop_loss_percent = params.get('stopLossPercent', 2.0) / 100
        self.take_profit_percent = params.get('takeProfitPercent', 4.0) / 100
        self.partial_exit_percent = params.get('partialExitPercent', 1.0) / 100
        self.trailing_stop_percent = params.get('trailingStopPercent', 0.8) / 100
        self.partial_exit_size = params.get('partialExitSize', 0.5)  # Exit 50% at first target
    
    def calculate_exit_levels(self, position: Position) -> Position:
        """
        Calculate all exit levels for a position
        
        Args:
            position: Position object
            
        Returns:
            Updated position with exit levels
        """
        if position.direction == 1:  # Long position
            position.initial_stop = position.entry_price * (1 - self.stop_loss_percent)
            position.partial_exit_target = position.entry_price * (1 + self.partial_exit_percent)
            position.final_exit_target = position.entry_price * (1 + self.take_profit_percent)
            position.trailing_stop = position.entry_price * (1 - self.trailing_stop_percent)
            position.highest_price = position.entry_price
        else:  # Short position
            position.initial_stop = position.entry_price * (1 + self.stop_loss_percent)
            position.partial_exit_target = position.entry_price * (1 - self.partial_exit_percent)
            position.final_exit_target = position.entry_price * (1 - self.take_profit_percent)
            position.trailing_stop = position.entry_price * (1 + self.trailing_stop_percent)
            position.lowest_price = position.entry_price
        
        return position
    
    def update_trailing_stop(self, position: Position, current_price: float) -> Position:
        """
        Update trailing stop based on current price
        
        Args:
            position: Position object
            current_price: Current market price
            
        Returns:
            Updated position with new trailing stop
        """
        if position.direction == 1:  # Long position
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
                # Update trailing stop
                position.trailing_stop = position.highest_price * (1 - self.trailing_stop_percent)
                # Ensure trailing stop doesn't go below initial stop
                position.trailing_stop = max(position.trailing_stop, position.initial_stop)
        else:  # Short position
            # Update lowest price
            if current_price < position.lowest_price:
                position.lowest_price = current_price
                # Update trailing stop
                position.trailing_stop = position.lowest_price * (1 + self.trailing_stop_percent)
                # Ensure trailing stop doesn't go above initial stop
                position.trailing_stop = min(position.trailing_stop, position.initial_stop)
        
        return position
    
    def check_exit(self, position: Position, bar: pd.Series) -> Tuple[bool, str, float]:
        """
        Check if any exit condition is met
        
        Args:
            position: Position object
            bar: Current bar data (must have 'high', 'low', 'close', 'entry_signal')
            
        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        if position.direction == 1:  # Long position
            # Check stop loss
            if low <= position.initial_stop and not position.partial_exit_filled:
                return True, 'stop_loss', position.initial_stop
            
            # Check trailing stop (after partial exit)
            if position.partial_exit_filled and low <= position.trailing_stop:
                return True, 'trailing_stop', position.trailing_stop
            
            # Check partial exit target
            if not position.partial_exit_filled and high >= position.partial_exit_target:
                return True, 'partial_exit', position.partial_exit_target
            
            # Check final exit target
            if position.partial_exit_filled and high >= position.final_exit_target:
                return True, 'final_target', position.final_exit_target
            
            # Check signal reversal
            if 'entry_signal' in bar and bar['entry_signal'] == -1:
                return True, 'signal_reversal', close
            
            # Check score weakening
            if 'score_weakening' in bar and bar['score_weakening']:
                return True, 'score_weakening', close
        
        else:  # Short position
            # Check stop loss
            if high >= position.initial_stop and not position.partial_exit_filled:
                return True, 'stop_loss', position.initial_stop
            
            # Check trailing stop (after partial exit)
            if position.partial_exit_filled and high >= position.trailing_stop:
                return True, 'trailing_stop', position.trailing_stop
            
            # Check partial exit target
            if not position.partial_exit_filled and low <= position.partial_exit_target:
                return True, 'partial_exit', position.partial_exit_target
            
            # Check final exit target
            if position.partial_exit_filled and low <= position.final_exit_target:
                return True, 'final_target', position.final_exit_target
            
            # Check signal reversal
            if 'entry_signal' in bar and bar['entry_signal'] == 1:
                return True, 'signal_reversal', close
            
            # Check score weakening
            if 'score_weakening' in bar and bar['score_weakening']:
                return True, 'score_weakening', close
        
        return False, None, None
    
    def process_partial_exit(self, position: Position) -> Position:
        """
        Process partial exit
        
        Args:
            position: Position object
            
        Returns:
            Updated position
        """
        position.partial_exit_filled = True
        position.remaining_size = position.size * (1 - self.partial_exit_size)
        return position


def simulate_exits(df: pd.DataFrame, positions: List[Position], params: Dict) -> pd.DataFrame:
    """
    Simulate exit management for all positions
    
    Args:
        df: DataFrame with price data and signals
        positions: List of Position objects
        params: Strategy parameters
        
    Returns:
        DataFrame with exit signals
    """
    exit_manager = ExitManager(params)
    result = df.copy()
    
    # Initialize exit columns
    result['exit_signal'] = 0
    result['exit_reason'] = ''
    result['exit_price'] = np.nan
    result['exit_size'] = 0.0
    
    for position in positions:
        position = exit_manager.calculate_exit_levels(position)
        
        # Process each bar after entry
        for i in range(position.entry_bar + 1, len(result)):
            bar = result.iloc[i]
            
            # Update trailing stop
            position = exit_manager.update_trailing_stop(position, bar['close'])
            
            # Check for exit
            should_exit, exit_reason, exit_price = exit_manager.check_exit(position, bar)
            
            if should_exit:
                if exit_reason == 'partial_exit':
                    # Partial exit
                    result.loc[result.index[i], 'exit_signal'] = position.direction
                    result.loc[result.index[i], 'exit_reason'] = exit_reason
                    result.loc[result.index[i], 'exit_price'] = exit_price
                    result.loc[result.index[i], 'exit_size'] = position.size * exit_manager.partial_exit_size
                    
                    # Update position
                    position = exit_manager.process_partial_exit(position)
                else:
                    # Full or remaining exit
                    result.loc[result.index[i], 'exit_signal'] = position.direction
                    result.loc[result.index[i], 'exit_reason'] = exit_reason
                    result.loc[result.index[i], 'exit_price'] = exit_price
                    result.loc[result.index[i], 'exit_size'] = position.remaining_size
                    
                    # Position closed
                    break
    
    return result


def calculate_exit_performance(entry_price: float, exit_price: float, 
                               direction: int, size: float = 1.0) -> Dict:
    """
    Calculate performance metrics for an exit
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        direction: Position direction (1 for long, -1 for short)
        size: Position size
        
    Returns:
        Dictionary with performance metrics
    """
    if direction == 1:  # Long
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl = (exit_price - entry_price) * size
    else:  # Short
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        pnl = (entry_price - exit_price) * size
    
    return {
        'pnl': pnl,
        'pnl_percent': pnl_pct,
        'r_multiple': pnl_pct / 2.0 if direction == 1 else pnl_pct / 2.0  # Assuming 2% initial risk
    }


def analyze_exit_reasons(df: pd.DataFrame) -> Dict:
    """
    Analyze distribution of exit reasons
    
    Args:
        df: DataFrame with exit signals
        
    Returns:
        Dictionary with exit reason statistics
    """
    exit_data = df[df['exit_signal'] != 0].copy()
    
    if len(exit_data) == 0:
        return {}
    
    # Count exit reasons
    exit_counts = exit_data['exit_reason'].value_counts().to_dict()
    
    # Calculate win rates by exit reason
    exit_data['is_winner'] = exit_data.apply(
        lambda row: calculate_exit_performance(
            row.get('entry_price', row['close']),
            row['exit_price'],
            row['exit_signal']
        )['pnl_percent'] > 0,
        axis=1
    )
    
    win_rates = exit_data.groupby('exit_reason')['is_winner'].mean().to_dict()
    
    # Average PnL by exit reason
    exit_data['pnl_pct'] = exit_data.apply(
        lambda row: calculate_exit_performance(
            row.get('entry_price', row['close']),
            row['exit_price'],
            row['exit_signal']
        )['pnl_percent'],
        axis=1
    )
    
    avg_pnl = exit_data.groupby('exit_reason')['pnl_pct'].mean().to_dict()
    
    return {
        'counts': exit_counts,
        'win_rates': win_rates,
        'avg_pnl': avg_pnl
    }


def get_active_exit_levels(position: Position, params: Dict) -> Dict:
    """
    Get current exit levels for an active position
    
    Args:
        position: Position object
        params: Strategy parameters
        
    Returns:
        Dictionary with current exit levels
    """
    exit_manager = ExitManager(params)
    position = exit_manager.calculate_exit_levels(position)
    
    return {
        'initial_stop': position.initial_stop,
        'partial_exit_target': position.partial_exit_target,
        'final_exit_target': position.final_exit_target,
        'trailing_stop': position.trailing_stop,
        'partial_exit_filled': position.partial_exit_filled,
        'remaining_size': position.remaining_size
    }
