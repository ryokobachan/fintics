import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class BopStrategy(Strategy):
    def __init__(self, dataframe, trial=None):
        df = dataframe.copy()
        df['Y'] = Indicator.BOP(df['Open'], df['High'], df['Low'], df['Close'])
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
