import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MovingAverageCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=7, t2=14, ma_type=2, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.ma_t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', t1, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        df['Y'] = (df[self.PRICE].MA(t1, ma_type=ma_type) - df[self.PRICE].MA(t2, ma_type=ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
