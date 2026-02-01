import pandas as pd
from fintics.strategy.strategy import Strategy, OptimizeStrategy
from fintics.indicator import Indicator

class MlStrategy(Strategy):
    _special = True
    def __init__(self, df: pd.DataFrame, model, datasets:list=[]):
        _df = df.copy()
        for dataset in datasets:
            _df = dataset(_df)
        df['y'] = model.predict(_df)
        super().__init__(df)
