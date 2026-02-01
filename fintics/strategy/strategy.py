"""Collection of trading strategy implementations."""

import numpy as np
import pandas as pd
from fintics.indicator import Indicator

class Strategy(pd.Series):

    ma_t_min = 1
    t_min = 2
    t_max = 250
    t_step = 1

    ma_min = 0
    ma_max1 = 9
    ma_max2 = 8
    talib_ma = [0, 1, 2, 3, 4, 5, 6, 8] # exclude 7
    ma_step = 1

    BUY_POSITION = 1
    KEEP_POSITION = 0
    SELL_POSITION = -1

    PRICE = 'Close'

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Strategy with a DataFrame containing a 'y' column.
        
        Args:
            df: pd.DataFrame with a 'y' column representing trading signals
        
        Raises:
            TypeError: If df is not a DataFrame
            ValueError: If df doesn't contain a 'y' column
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
        if 'y' not in df.columns:
            raise ValueError("DataFrame must contain a 'y' column")
        
        super().__init__(df['y'])
        self._df = df


    def getY(self, only_buy:bool=True, reverse:bool=False):
        y = self.copy()

        if reverse:
            y = y * (-1)
        
        # only long position
        if only_buy:
            y = y.mask(y < 0, 0)

        return y
    
    def getLastY(self, only_buy:bool=True):
        y = self.getY(only_buy)
        return y.iloc[-1]

class OptimizeStrategy(Strategy):
    _special = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class AverageStrategy(Strategy):
    _special = True
    def __init__(self, df, strategies=[], only_buy=True):
        if len(strategies) == 0:
            raise Exception('strategies cannot be zero length list.')
        d = df.copy().assign(y=0)
        for st in strategies:
            y = st['strategy'](df, **st['params'])
            if only_buy:
                y.loc[y < 0] = 0
            d['y'] += y
        d['y'] /= len(strategies)
        super().__init__(df)

class CustomStrategy(Strategy):
    _special = True
    def __init__(self, df, strategy):
        '''
        def strategy(self, df, index: int):
            
            # write process

            if CONDITION_1:
                y = self.BUY_POSITION

            if CONDITION_2:
                y = self.SELF_POSITION

            y = self.KEEP_POSITION
        '''
        d = df.reset_index().assign(y=0).copy()
        for index, _ in d.iterrows():
            d.loc[index, 'y'] = strategy(d, index)
        df['y'] = pd.Series(d['y'].values, index=df.index)
        super().__init__(df)

class KeepPositionStrategy(Strategy):
    def __init__(self, df):
        df['y'] = 1
        super().__init__(df)
