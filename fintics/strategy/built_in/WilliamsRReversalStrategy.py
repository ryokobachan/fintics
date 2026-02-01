import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

from .util import CALC_OVERLINE_OSCILLATOR_Y

class WilliamsRReversalStrategy(OptimizeStrategy):
    def __init__(self, df, t=14, signal=30, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            signal = trial.suggest_float('signal', 1, 100, step=0.1)

        df['Y'] = CALC_OVERLINE_OSCILLATOR_Y(Indicator.WILLR(df['High'], df['Low'], df['Close'], t=t) + 50, signal)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
