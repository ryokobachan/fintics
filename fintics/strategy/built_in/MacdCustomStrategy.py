import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MacdCustomStrategy(OptimizeStrategy):
    def __init__(self, df, fast=12, slow=26, signal=9, ma_type=4, trial=None):
        if trial:
            fast = trial.suggest_int('fast', self.t_min, self.t_max, step=self.t_step)
            slow = trial.suggest_int('slow', fast, self.t_max, step=self.t_step)
            signal = trial.suggest_int('signal', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max1, self.ma_step)))
        macd = df[self.PRICE].MA(fast, ma_type=ma_type) - df[self.PRICE].MA(slow, ma_type=ma_type)
        sign = macd.MA(signal, ma_type=ma_type)
        df['Y'] = (macd - sign)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
