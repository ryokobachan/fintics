import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class KalmanFilterChangeStrategy(OptimizeStrategy):
    def __init__(self, df, t=14, trial=None):
        if trial:
            t = trial.suggest_float('t', 1e-4, 1e12, log=True)
        df['Y'] = Indicator.KalmanMA(df[self.PRICE], t=t).diff()
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
