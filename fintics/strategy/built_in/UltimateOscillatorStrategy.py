import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class UltimateOscillatorStrategy(Strategy):
    def __init__(self, dataframe, t1=7, t2=14, t3=28, trial=None):
        df = dataframe.copy()
        if trial:
            t1 = trial.suggest_int('t1', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
            t2 = trial.suggest_int('t2', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
            t3 = trial.suggest_int('t3', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
        df['Y'] = (Indicator.ULTOSC(df['High'], df['Low'], df['Close'], t1=t1, t2=t2, t3=t3) - 50)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
