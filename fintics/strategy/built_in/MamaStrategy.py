import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MamaStrategy(OptimizeStrategy):
    def __init__(self, df, fastlimit=0.5, slowlimit=0.05, trial=None):
        if trial:
            fastlimit = trial.suggest_float('fastlimit', 0.01, 0.99, step=0.001)
            slowlimit = trial.suggest_float('slowlimit', 0.01, fastlimit, step=0.001)  
        mama = Indicator.MAMA(df[self.PRICE], fastlimit=fastlimit, slowlimit=slowlimit)
        df['Y'] = (mama[0] - mama[1])
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
