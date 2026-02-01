import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class IchimokuCloudStrategy(OptimizeStrategy):
    def __init__(self, df, t1=9, t2=26, t3=52, trial=None):
        if trial:
            t3 = trial.suggest_int('t3', self.t_min, self.t_max, step=self.t_step)
            t2 = trial.suggest_int('t2', self.t_min, t3, step=self.t_step)
            t1 = trial.suggest_int('t1', self.t_min, t2, step=self.t_step)
        df = Indicator.Ichimoku2(df, t1, t2, t3)
        df['y'] = 0

        # Three bullish signals in Ichimoku
        buy_condition1 = df['conversion_line'] > df['base_line']
        buy_condition2 = df['Low'] > df[['leading_span1', 'leading_span2']].max(axis=1)
        buy_condition3 = df['Close'] > df['High'].shift(t2)
        df.loc[buy_condition1 & buy_condition2 & buy_condition3, 'y'] = 1

        # Three bearish signals in Ichimoku
        sell_condition1 = df['conversion_line'] < df['base_line']
        sell_condition2 = df['High'] < df[['leading_span1', 'leading_span2']].min(axis=1)
        sell_condition3 = df['Close'] < df['Low'].shift(t2)
        df.loc[sell_condition1 & sell_condition2 & sell_condition3, 'y'] = -1

        super().__init__(df)
