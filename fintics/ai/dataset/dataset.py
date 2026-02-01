"""Dataset classes that generate features and targets for models."""

from typing import Literal, Optional
import pandas as pd
import talib
from fintics.indicator import Indicator


class Dataset(pd.DataFrame):
    def __init__(self, df):
        super().__init__(df)

class BasicDataset():

    ma_t_min = 1
    t_min = 2
    t_max = 250
    t_step = 1

    ma_min = 0
    ma_max1 = 9
    ma_max2 = 8
    ma_step = 1

    class Basic(Dataset):
        def __init__(self, df):
            df['HL'] = (df.High + df.Low) / 2
            df['OC'] = (df.Open + df.Close) / 2
            df['HLC'] = (df.High + df.Low + df.Close) / 3
            df['HLCC'] = (df.High + df.Low + df.Close * 2) / 4
            df['HLO'] = (df.High + df.Low + df.Open.shift(-1)) / 3
            df['HLOO'] = (df.High + df.Low  + df.Open.shift(-1) * 2) / 4
            df['HLHL'] = ((df.High + df.Low) + (df.High + df.Low).shift(-1)) / 4
            df['OCOC'] = ((df.Open + df.Close) + (df.Open + df.Close).shift(-1)) / 4
            df['HL_rolling'] = (df.High.rolling(2).max() + df.Low.rolling(2).min()) / 2
            df['OpenPL'] = df.Open.diff().shift(-2)
            df['ClosePL'] = df.Close.diff().shift(-1)
            super().__init__(df)


    class X_ml(Dataset):
        def __init__(self, df, price_type='Close', t_start=1, t_end=11, t_step=1):
            for t in range(t_start, t_end+1, t_step):
                df[f'X_Close-diff_{t}'] = df[price_type].pct_change().MA(t, ma_type=0)
                df[f'X_HL-C_{t}'] = (df.Close - (df.High + df.Low).MA(t, ma_type=0) / 2) / df.Close
                df[f'X_HLC-diff_{t}'] = ((df.High + df.Low + df.Close) / 3).pct_change(t)
                df[f'X_RSI_{t}'] = df[price_type].RSI(t+1)
                df[f'X_RSI-diff_{t}'] = df[price_type].RSI(t+1).diff()
                df[f'X_SMA-EMA_{t}'] = df[price_type].EMA(t+1) / df[price_type].SMA(t+1)
                df[f'X_EMA-TSF_{t}'] = df[price_type].TSF(t+1) / df[price_type].EMA(t+1)
                df[f'X_TSF_{t}'] = df[price_type].TSF(t+1).pct_change()
                df[f'X_EMA-CROSS_{t}'] = df[price_type] / df[price_type].EMA(t+1)
                df[f'X_DMI-diff_{t}'] = (Indicator.PLUS_DI(df.High, df.Low, df.Close, t+1) - Indicator.MINUS_DI(df.High, df.Low, df.Close, t+1)).diff()
                df[f'X_MOM-diff_{t}'] = Indicator.MOM(df[price_type], t+1).diff()
                df[f'X_WILLR_{t}'] = Indicator.WILLR(df['High'], df['Low'], df['Close'], t=t+1)

            super().__init__(df)
    
    class X_all(Dataset):
        def __init__(self, df, t_start=1, t_end=11, t_step=1):
            o = df['Open']
            h = df['High']
            l = df['Low']
            c = df['Close']
            volume = df['Volume']
            
            orig_columns = df.columns

            hilo = (h + l) / 2

            df[f'X_AD'] = talib.AD(h, l, c, volume) / c
            df[f'X_HT_PHASOR_inphase'], df[f'X_HT_PHASOR_quadrature'] = talib.HT_PHASOR(c)
            df[f'X_HT_PHASOR_inphase'] /= c
            df[f'X_HT_PHASOR_quadrature'] /= c
            df[f'X_HT_TRENDLINE'] = (talib.HT_TRENDLINE(c) - hilo) / c
            df[f'X_OBV'] = talib.OBV(c, volume) / c
            df[f'X_BOP'] = talib.BOP(o, h, l, c)
            df[f'X_TRANGE'] = talib.TRANGE(h, l, c) / c
            df[f'X_HT_DCPERIOD'] = talib.HT_DCPERIOD(c)
            df[f'X_HT_DCPHASE'] = talib.HT_DCPHASE(c)
            df[f'X_HT_SINE_sine'], df[f'X_HT_SINE_leadsine'] = talib.HT_SINE(c)
            df[f'X_HT_TRENDMODE'] = talib.HT_TRENDMODE(c)

            for t1 in range(t_start, t_end+1, t_step):
            # Normalize by subtracting hilo or close price and dividing by close
                df[f'X_BBANDS_upperband_{t1}'], df[f'X_BBANDS_middleband_{t1}'], df[f'X_BBANDS_lerband_{t1}'] = talib.BBANDS(c, timeperiod=t1, nbdevup=2, nbdevdn=2, matype=0)
                df[f'X_BBANDS_upperband_{t1}'] = (df[f'X_BBANDS_upperband_{t1}'] - hilo) / c
                df[f'X_BBANDS_middleband_{t1}'] = (df[f'X_BBANDS_middleband_{t1}'] - hilo) / c
                df[f'X_BBANDS_lerband_{t1}'] = (df[f'X_BBANDS_lerband_{t1}'] - hilo) / c
                df[f'X_DEMA_{t1}'] = (talib.DEMA(c, timeperiod=t1) - hilo) / c
                df[f'X_EMA_{t1}'] = (talib.EMA(c, timeperiod=t1) - hilo) / c
                df[f'X_KAMA_{t1}'] = (talib.KAMA(c, timeperiod=t1) - hilo) / c
                df[f'X_MA_{t1}'] = (talib.MA(c, timeperiod=t1, matype=0) - hilo) / c
                df[f'X_MIDPOINT_{t1}'] = (talib.MIDPOINT(c, timeperiod=t1) - hilo) / c
                df[f'X_SMA_{t1}'] = (talib.SMA(c, timeperiod=t1) - hilo) / c
                df[f'X_T3_{t1}'] = (talib.T3(c, timeperiod=t1, vfactor=0) - hilo) / c
                df[f'X_TEMA_{t1}'] = (talib.TEMA(c, timeperiod=t1) - hilo) / c
                df[f'X_TRIMA_{t1}'] = (talib.TRIMA(c, timeperiod=t1) - hilo) / c
                df[f'X_WMA_{t1}'] = (talib.WMA(c, timeperiod=t1) - hilo) / c
                df[f'X_LINEARREG_{t1}'] = (talib.LINEARREG(c, timeperiod=t1) - c) / c
                df[f'X_LINEARREG_INTERCEPT_{t1}'] = (talib.LINEARREG_INTERCEPT(c, timeperiod=t1) - c) / c


                # Normalize by dividing by close price
                
                
                df[f'X_LINEARREG_SLOPE_{t1}'] = talib.LINEARREG_SLOPE(c, timeperiod=t1) / c
                
                df[f'X_MINUS_DM_{t1}'] = talib.MINUS_DM(h, l, timeperiod=t1) / c
                df[f'X_MOM_{t1}'] = talib.MOM(c, timeperiod=t1) / c
                
                df[f'X_PLUS_DM_{t1}'] = talib.PLUS_DM(h, l, timeperiod=t1) / c
                df[f'X_STDDEV_{t1}'] = talib.STDDEV(c, timeperiod=t1, nbdev=1) / c


                df[f'X_ADX_{t1}'] = talib.ADX(h, l, c, timeperiod=t1)
                df[f'X_ADXR_{t1}'] = talib.ADXR(h, l, c, timeperiod=t1)
                df[f'X_AROON_aroondown_{t1}'], df[f'X_AROON_aroonup_{t1}'] = talib.AROON(h, l, timeperiod=t1)
                df[f'X_AROONOSC_{t1}'] = talib.AROONOSC(h, l, timeperiod=t1)
                
                df[f'X_CCI_{t1}'] = talib.CCI(h, l, c, timeperiod=t1)
                df[f'X_DX_{t1}'] = talib.DX(h, l, c, timeperiod=t1)
                # Skip MACDEXT and MACDFIX as they are likely equivalent
                df[f'X_MFI_{t1}'] = talib.MFI(h, l, c, volume, timeperiod=t1)
                df[f'X_MINUS_DI_{t1}'] = talib.MINUS_DI(h, l, c, timeperiod=t1)
                df[f'X_PLUS_DI_{t1}'] = talib.PLUS_DI(h, l, c, timeperiod=t1)
                df[f'X_RSI_{t1}'] = talib.RSI(c, timeperiod=t1)

                df[f'X_TRIX_{t1}'] = talib.TRIX(c, timeperiod=t1)
                
                df[f'X_WILLR_{t1}'] = talib.WILLR(h, l, c, timeperiod=t1)

                df[f'X_ATR_{t1}'] = talib.ATR(h, l, c, timeperiod=t1)
                df[f'X_NATR_{t1}'] = talib.NATR(h, l, c, timeperiod=t1)



                df[f'X_BETA_{t1}'] = talib.BETA(h, l, timeperiod=t1)
                df[f'X_CORREL_{t1}'] = talib.CORREL(h, l, timeperiod=t1)

                df[f'X_LINEARREG_ANGLE_{t1}'] = talib.LINEARREG_ANGLE(c, timeperiod=t1)

            super().__init__(df)
        
    class y_basic(Dataset):
        def __init__(self, df, price_type='Close', t=1, binary=True, trial=None):
            if trial:
                t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
            df['y'] = (df[price_type].shift(-t) - df[price_type]).STEP_MULTI()
            df['y'].loc[df['y'] == 0] = 1
            if binary:
                df['y'] = df['y'].STEP()
            super().__init__(df)
        
    class y_ma(Dataset):
        def __init__(self, df, price_type='Close', t_ma=2, t_shift=1, ma_type=0, binary=True, trial=None):
            if trial:
                t_ma = trial.suggest_int('t_ma', self.t_min, self.t_max, step=self.t_step)
                t_shift = trial.suggest_int('t_shift', 0, self.t_max, step=self.t_step)
                ma_type = trial.suggest_int('ma_type', self.ma_min, self.ma_max1, step=self.ma_step)
            df['y'] = df[price_type].MA(t_ma, ma_type=ma_type).diff().shift(-t_shift).STEP_MULTI()
            if binary:
                df['y'] = df['y'].STEP()
            super().__init__(df)
    
    class y_ma_simple(Dataset):
        def __init__(self, df, price_type='Close', t=2, ma_type=0, binary=True, trial=None):
            if trial:
                t = trial.suggest_int('t', self.t_min, self.t_max, step=self.t_step)
                ma_type = trial.suggest_int('ma_type', self.ma_min, self.ma_max1, step=self.ma_step)
            df['y'] = df[price_type].MA(t, ma_type=ma_type).diff().shift(-t).STEP_MULTI()
            if binary:
                df['y'] = df['y'].STEP()
            super().__init__(df)
        
    class y_zigzag(Dataset):
        def __init__(self, df, percentage=0.05, binary=True, trial=None):
            if trial:
                percentage = trial.suggest_float('percentage', 0.001, 0.250, step=0.001)
            df['y'] = Indicator.Zigzag(df, percentage=percentage).shift(-1).diff().STEP_MULTI()
            if binary:
                df['y'] = df['y'].STEP()
            super().__init__(df)
        
    class Reward(Dataset):
        def __init__(self, df, price_type: Literal['Close', 'Open']='Open'):
            c = df['Close'] if price_type == 'Close' else df['Open'].shift(-1)
            df['Reward'] = c.diff().shift(-1)
            super().__init__(df)
