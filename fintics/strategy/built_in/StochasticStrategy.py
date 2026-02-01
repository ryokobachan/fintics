import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class StochasticStrategy(OptimizeStrategy):
    def __init__(self, df, t_k=5, t_d=3, ma_type=0, trial=None):
        '''
        `signal`: 0~50
        ex) signal=30 -> entry_point: under 20 or over 80
        '''
        if trial:
            t_k = trial.suggest_int('t_k', self.t_min, self.t_max, step=self.t_step)
            t_d = trial.suggest_int('t_d', self.t_min, self.t_max, step=self.t_step)
            ma_type = trial.suggest_categorical('ma_type', list(range(self.ma_min, self.ma_max2, self.ma_step)))
        fastk, fastd = Indicator.STOCHF(df['High'], df['Low'], df['Close'], t_k, t_d, ma_type)
        df['Y'] = (fastk - fastd)
        df['y'] = df['Y'].STEP_MULTI()
        super().__init__(df)
