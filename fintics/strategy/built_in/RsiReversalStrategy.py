import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

from .util import CALC_OVERLINE_OSCILLATOR_Y

class RsiReversalStrategy(OptimizeStrategy):
    def __init__(self, df, t=14, signal=20, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            signal = trial.suggest_float('signal', 1, 50, step=0.1)

        df['Y'] = CALC_OVERLINE_OSCILLATOR_Y(Indicator.RSI(df[self.PRICE], t=t), signal)
        df['y'] = df['Y'].STEP_MULTI()
        
        super().__init__(df)
