import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MlCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t=12, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
        df['Y'] = (df[self.PRICE].TSF(t) - df[self.PRICE].SMA(t))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
