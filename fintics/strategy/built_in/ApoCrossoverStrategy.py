import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class ApoCrossoverStrategy(OptimizeStrategy):
    def __init__(self, dataframe, t_fast1=9, t_slow1=18, t_fast2=12, t_slow2=26, ma_type=0, trial=None):
        df = dataframe.copy()
        if trial:
            t_slow1 = trial.suggest_int('t_slow1', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
            t_fast1 = trial.suggest_int('t_fast1', Strategy.t_min, t_slow1, step=Strategy.t_step)
            t_slow2 = trial.suggest_int('t_slow2', Strategy.t_min, Strategy.t_max, step=Strategy.t_step)
            t_fast2 = trial.suggest_int('t_fast2', Strategy.t_min, t_slow2, step=Strategy.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max2, self.ma_step)))
        apo1 = Indicator.APO(df[self.PRICE], fastperiod=t_fast1, slowperiod=t_slow1, ma_type=ma_type)
        apo2 = Indicator.APO(df[self.PRICE], fastperiod=t_fast2, slowperiod=t_slow2, ma_type=ma_type)
        df['Y'] = (apo1 - apo2)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
