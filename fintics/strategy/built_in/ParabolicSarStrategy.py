import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class ParabolicSarStrategy(OptimizeStrategy):
    def __init__(self, df, acceleration=0.02, maximum=0.2, trial=None):
        if trial:
            acceleration = trial.suggest_float('acceleration', 0.001, 1, step=0.001)
            maximum = trial.suggest_float('maximum', acceleration, 1, step=0.001)
        df['Y'] = (df[self.PRICE] - Indicator.SAR(df['High'], df['Low'], acceleration=acceleration, maximum=maximum))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
