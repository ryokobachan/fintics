import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class PpoStrategy(OptimizeStrategy):
    def __init__(self, dataframe, fast=12, slow=26, ma_type=0, trial=None):
        df = dataframe.copy()
        if trial:
            fast = trial.suggest_int('fast', self.t_min, self.t_max, step=self.t_step)
            slow = trial.suggest_int('slow', fast, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max2, self.ma_step)))
        df['Y'] = Indicator.PPO(df[self.PRICE], fastperiod=fast, slowperiod=slow, ma_type=ma_type)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
