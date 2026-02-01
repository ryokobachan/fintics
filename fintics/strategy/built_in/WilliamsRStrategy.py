import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class WilliamsRStrategy(OptimizeStrategy):
    def __init__(self, df, t=14, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)

        df['Y'] = Indicator.WILLR(df['High'], df['Low'], df['Close'], t=t) + 50
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
