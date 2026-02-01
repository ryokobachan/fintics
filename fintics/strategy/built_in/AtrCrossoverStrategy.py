import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class AtrCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=14, t2=25, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, self.t_max, step=self.t_step)
        df['Y'] = (Indicator.ATR(df.High, df.Low, df.Close, t1) - Indicator.ATR(df.High, df.Low, df.Close, t2))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
