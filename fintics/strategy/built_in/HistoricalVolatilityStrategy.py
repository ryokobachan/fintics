import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class HistoricalVolatilityStrategy(OptimizeStrategy):
    def __init__(self, df, t1=14, t2=25, ma_type=0, trial=None):
        if trial:
            t1 = trial.suggest_int('t1', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        df['Y'] = (Indicator.HV(df.Close, t1) - Indicator.HV(df.Close, t1).MA(t2, ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
