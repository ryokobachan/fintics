"""
Custom Strategy Template

This template provides a comprehensive starting point for creating custom trading strategies.
It demonstrates how to combine multiple indicators and implement optimization support.
"""

import pandas as pd
from fintics.strategy import Strategy
from fintics.indicator import Indicator


class STRATEGY_NAME(Strategy):
    """
    Custom Strategy Template
    
    This template shows how to combine multiple indicators for trading signals.
    
    Parameters:
        df: pd.DataFrame - OHLC data
        rsi_period: int - RSI period (default: 14)
        ma_short: int - Short MA period (default: 10)
        ma_long: int - Long MA period (default: 30)
        trial: Optuna trial object for optimization (optional)
        reverse: bool - Reverse signals (default: False)
    """
    
    def __init__(self, df: pd.DataFrame, rsi_period: int = 14, ma_short: int = 10, 
                 ma_long: int = 30, trial=None, reverse: bool = False):
        """
        Initialize Strategy
        
        Args:
            df: DataFrame with OHLC data
            rsi_period: RSI time period
            ma_short: Short moving average period
            ma_long: Long moving average period
            trial: Optuna trial for optimization (optional)
            reverse: Whether to reverse signals
        """
        # If trial is provided (during optimization), suggest parameter ranges
        if trial:
            rsi_period = trial.suggest_int('rsi_period', 5, 30)
            ma_short = trial.suggest_int('ma_short', 5, 20)
            ma_long = trial.suggest_int('ma_long', 20, 50)
        
        # Calculate indicators
        rsi = Indicator.RSI(df['Close'], timeperiod=rsi_period)
        ma_short_values = Indicator.SMA(df['Close'], timeperiod=ma_short)
        ma_long_values = Indicator.SMA(df['Close'], timeperiod=ma_long)
        
        # Generate signals based on multiple conditions
        df['y'] = 0
        
        # Buy signal: MA crossover (short crosses above long) AND RSI < 70
        buy_condition = (
            (ma_short_values > ma_long_values) & 
            (ma_short_values.shift(1) <= ma_long_values.shift(1)) & 
            (rsi < 70)
        )
        df.loc[buy_condition, 'y'] = 1
        
        # Sell signal: MA crossover (short crosses below long) OR RSI > 70
        sell_condition = (
                (ma_short_values < ma_long_values) & 
                (ma_short_values.shift(1) >= ma_long_values.shift(1)
            ) | (rsi > 70)
        )
        df.loc[sell_condition, 'y'] = -1
        
        # Initialize parent Strategy class
        super().__init__(df)
