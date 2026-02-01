import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class AtrChangeStrategy(OptimizeStrategy):
    def __init__(self, df, t=84, t_diff=50, trial=None):
        self.t_max = 250
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            t_diff = trial.suggest_int('t_diff', self.t_min, self.t_max, step=self.t_step)
        df['Y'] = Indicator.ATR(df.High, df.Low, df.Close, t).diff(t_diff)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
