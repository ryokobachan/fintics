import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class RciStrategy(OptimizeStrategy):
    def __init__(self, df, t=9, signal=50, trial=None):
        if trial:
            t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            signal = trial.suggest_float('signal', 1, 50, step=0.1)
        df['Y'] = Indicator.RCI(df['Close'], t)
        df['y'] = df['Y'].STEP_MULTI(signal)
        super().__init__(df)
