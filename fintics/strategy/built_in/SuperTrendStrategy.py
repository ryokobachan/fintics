import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class SuperTrendStrategy(Strategy):
    def __init__(self, df, t=14, factor=3, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            factor = trial.suggest_float('factor', 0.1, 5, step=0.1)
        df['y'] = Indicator.SUPERTREND(df['High'], df['Low'], df['Close'], t=t, factor=factor, only_result=True)
        super().__init__(df)
