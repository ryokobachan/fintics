import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class HeikinAshiStrategy(OptimizeStrategy):
    def __init__(self, df, t_shift=0, t_ma=1, ma_type=0, trial=None):
        if trial:
            t_shift = trial.suggest_int('t_shift', 0, 14, step=self.t_step)
            t_ma = trial.suggest_int('t_ma', 1, 14, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max2, self.ma_step)))
        df_h = df.Heikinashi(t_shift=t_shift, t_ma=t_ma, ma_type=ma_type)
        df['Y'] = (df_h.Close - df_h.Open)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
