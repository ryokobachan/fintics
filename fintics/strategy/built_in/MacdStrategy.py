import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MacdStrategy(OptimizeStrategy):
    def __init__(self, df, fast=12, slow=26, signal=9, trial=None):
        if trial:
            fast = trial.suggest_int('fast', self.t_min, self.t_max, step=self.t_step)
            slow = trial.suggest_int('slow', fast, self.t_max, step=self.t_step)
            signal = trial.suggest_int('signal', self.t_min, self.t_max, step=self.t_step)
        macd = df[self.PRICE].EMA(fast) - df[self.PRICE].EMA(slow)
        sign = macd.SMA(signal)
        df['Y'] = macd - sign
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
