import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class UpDownCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=14, t2=30, ma_type=0, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        ud = df[self.PRICE].diff().STEP_MULTI().cumsum()
        df['Y'] = (ud.MA(t1, ma_type) - ud.MA(t2, ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
