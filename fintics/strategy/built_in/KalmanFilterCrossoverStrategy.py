import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class KalmanFilterCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t1=14, t2=25, trial=None):
        if trial:
            t1 = trial.suggest_float('t1', 1e-4, 1e12, log=True)
            t2 = trial.suggest_float('t2', 1e-4, 1e12, log=True)
        df['Y'] = Indicator.KalmanMA(df[self.PRICE], t=t1) - Indicator.KalmanMA(df[self.PRICE], t=t2)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
