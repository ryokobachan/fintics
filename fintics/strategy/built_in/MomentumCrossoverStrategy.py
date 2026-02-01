import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MomentumCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=8, t2=14, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', t1, self.t_max, step=self.t_step)
        df['Y'] = (df[self.PRICE].MOM(t2) - df[self.PRICE].MOM(t1))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
