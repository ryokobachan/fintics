import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class CmoStrategy(OptimizeStrategy):
    def __init__(self, dataframe, t=14, trial=None):
        df = dataframe.copy()
        if trial:
            t = trial.suggest_int('t', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
        df['Y'] = Indicator.CMO(df[self.PRICE], t=t)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
