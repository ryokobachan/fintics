import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class TripleMovingAverageStrategy(OptimizeStrategy):
    def __init__(self, df, ma_type=2, t_s=5, t_m=25, t_l=75, trial=None):
        if trial:
            t_s = trial.suggest_int('t_s', self.ma_t_min, self.t_max, step=self.t_step)
            t_m = trial.suggest_int('t_m', t_s, self.t_max, step=self.t_step)
            t_l = trial.suggest_int('t_l', t_m, self.t_max, step=self.t_step)

        df['Y1'] = df[self.PRICE].MA(t_s, ma_type=ma_type) - df[self.PRICE].MA(t_m, ma_type=ma_type)
        df['Y2'] = df[self.PRICE].MA(t_m, ma_type=ma_type) - df[self.PRICE].MA(t_l, ma_type=ma_type)

        df['y'] = (df['Y1'].STEP_MULTI() + df['Y2'].STEP_MULTI()) / 2
        super().__init__(df)
