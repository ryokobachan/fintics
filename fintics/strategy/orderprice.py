import numpy as np
import pandas as pd
from fintics.indicator import Indicator

class OrderPrice:
    ma_t_min = 1
    t_min = 2
    t_max = 250
    t_step = 1

    ma_min = 0
    ma_max = 6
    ma_step = 1
    
    def __init__(self, buy_price: pd.Series, sell_price: pd.Series):
        self.buy_price = buy_price
        self.sell_price = sell_price

    def __getitem__(self, index):
        if index == 0:
            return self.buy_price
        elif index == 1:
            return self.sell_price
        else:
            raise IndexError("Index out of range (0 or 1 only)")

class Marker_OrderPrice(OrderPrice):
    def __init__(self, _df, trial=None):
        super().__init__(_df['Open_buy'].shift(-1), _df['Open_sell'].shift(-1))

class ATR_OrderPrice(OrderPrice):
    def __init__(self, _df, o_delta = 100000, o_t_atr=10, o_t_ma=5, trial=None):
        if trial is not None:
            o_delta = trial.suggest_float("o_delta", 0.01, 10**3, log=True)
            o_t_ma = trial.suggest_int("o_t_ma", self.t_min, self.t_max)
            o_t_atr = trial.suggest_int("o_t_atr", self.t_min, self.t_max)
        _atr = Indicator.ATR(_df['High'].EMA(o_t_ma), _df['Low'].EMA(o_t_ma), _df['Close'].EMA(o_t_ma), t=o_t_atr)

        _buy_price = _df['Price'] - (_atr * o_delta)
        _sell_price = _df['Price'] + (_atr * o_delta)

        super().__init__(_buy_price, _sell_price)

class BBANDS_OrderPrice(OrderPrice):
    def __init__(self, _df, o_nbdev=10, o_t_bb=5, o_ma_type=0, trial=None):
        if trial is not None:
            o_nbdev = trial.suggest_float("o_nbdev", 0.001, 10.0, log=True)
            o_t_bb = trial.suggest_int("o_t_bb", self.t_min, self.t_max)
            o_ma_type = trial.suggest_int("o_ma_type", self.ma_min, self.ma_max)

        _sell_price, _, _buy_price = Indicator.BBANDS(_df['Close'], t=o_t_bb, nbdevdn=o_nbdev, nbdevup=o_nbdev, ma_type=o_ma_type)

        super().__init__(_buy_price, _sell_price)