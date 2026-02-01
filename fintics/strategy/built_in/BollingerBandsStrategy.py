import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class BollingerBandsStrategy(OptimizeStrategy):
    def __init__(self, df, t=4, std=2, ma_type=2, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            std = trial.suggest_float('std', -10, 10, step=0.1)
        ma_type = 2

        _delta = 1 if std >= 0 else -1
        df['Y'] = (df['Close'] - df['Close'].BOLLINGERBANDS(t, std, ma_type)) * _delta
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
