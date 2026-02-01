import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class CorrelationCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=10, t2=12, ma_type=6, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        CORREL = Indicator.CORREL(df['High'], df['Low'], t1)
        df['Y'] = (CORREL - CORREL.MA(t2, ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
