import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class AroonCrossoverStrategy(OptimizeStrategy):
    def __init__(self, dataframe, t1=14, t2=25, trial=None):
        df = dataframe.copy()
        if trial:
            t1 = trial.suggest_int('t1', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
            t2 = trial.suggest_int('t2', t1, Strategy.t_max, step=Strategy.t_step)
        df['Y'] = (Indicator.AROONOSC(df['High'], df['Low'], t=t1) - Indicator.AROONOSC(df['High'], df['Low'], t=t2))
        df['y'] = df['Y'].STEP_MULTI()

        super().__init__(df)
