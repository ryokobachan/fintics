import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class CorrelationMaCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t=14, t_ma_1=7, t_ma_2=14, ma_type=3, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            t_ma_1 = trial.suggest_int('t_ma_1', self.t_min, self.t_max, step=self.t_step)
            t_ma_2 = trial.suggest_int('t_ma_2', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        CORREL = Indicator.CORREL(df['High'], df['Low'], t)
        df['Y'] = (CORREL.MA(t_ma_1, ma_type) - CORREL.MA(t_ma_2, ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
