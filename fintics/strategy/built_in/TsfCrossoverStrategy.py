import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class TsfCrossoverStrategy(OptimizeStrategy):
    def __init__(self, df, t_tsf=14, t_ma=14, ma_type=0, trial=None):
        if trial:
            t_tsf = trial.suggest_int('t_tsf', self.t_min, self.t_max, step=self.t_step)
            t_ma = trial.suggest_int('t_ma', self.t_min, self.t_max, step=self.t_step)
        tsf = df[self.PRICE].TSF(t_tsf)
        df['Y'] = (tsf - tsf.MA(t_ma, ma_type=ma_type))
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
