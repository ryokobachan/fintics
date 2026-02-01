import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class WilliamsRCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=9, t2=14, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', t1, self.t_max, step=self.t_step)
        df['Y'] = (Indicator.WILLR(df['High'], df['Low'], df['Close'], t=t1) - Indicator.WILLR(df['High'], df['Low'], df['Close'], t=t2))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
