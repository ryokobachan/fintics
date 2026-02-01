import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

from .util import CALC_OVERLINE_OSCILLATOR_Y

class StochasticReversalStrategy(OptimizeStrategy):
    def __init__(self, df, t=5, signal=50, trial=None):
        '''
        `signal`: 0~50
        ex) signal=30 -> entry_point: under 20 or over 80
        '''
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            signal = trial.suggest_float('signal', 1, 100, step=0.1)
        stoch, _ = Indicator.STOCHF(df['High'], df['Low'], df['Close'], t)

        df['Y'] = CALC_OVERLINE_OSCILLATOR_Y(stoch, signal)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
