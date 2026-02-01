import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MfiStrategy(Strategy):
    def __init__(self, dataframe, t=14, trial=None):
        df = dataframe.copy()
        if trial:
            t = trial.suggest_int('t', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
        df['Y'] = Indicator.MFI(df['High'], df['Low'], df['Close'], df['Volume'], t=t)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
