import pandas as pd
from fintics.strategy.strategy import Strategy

def CALC_OVERLINE_OSCILLATOR_Y(oscillator: pd.Series, signal: float):
    H, L = (oscillator - (50 + signal)) / 100, ((50 - signal) - oscillator) / 100
    Y = Strategy.KEEP_POSITION + Strategy.BUY_POSITION * L.STEP() + Strategy.SELL_POSITION * H.STEP()
    Y = Y.zerona().ffill()
    Y = (Y > 0) * H - (Y < 0) * L
    return Y
