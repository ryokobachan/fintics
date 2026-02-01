import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class SmoothedAtrCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=14, t2=25, t_ma=7, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, self.t_max, step=self.t_step)
            t_ma = trial.suggest_int('t_ma', self.t_min, self.t_max, step=self.t_step)
        h, l = df.High.rolling(t_ma).max(), df.Low.rolling(t_ma).min()
        c = df.Close.rolling(t_ma).mean()
        df['Y'] = (Indicator.ATR(h, l, c, t1) - Indicator.ATR(h, l, c, t2))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
