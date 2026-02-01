"""Custom technical indicator implementations with dynamic talib delegation."""

import pandas as pd
import numpy as np
from numba import njit
import talib
from functools import wraps
from typing import Literal

# Decorator definitions for pandas extension registration
def seriesIndicator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__series_indicator__ = True
    return wrapper

def dataframeIndicator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__dataframe_indicator__ = True
    return wrapper


class IndicatorMeta(type):
    """Metaclass to enable dynamic delegation to talib."""
    
    def __getattr__(cls, name):
        """
        Dynamically delegate to talib if the method doesn't exist in Indicator.
        This allows calling any talib function through Indicator.XXX().
        """
        if hasattr(talib, name):
            talib_func = getattr(talib, name)
            
            @wraps(talib_func)
            def smart_wrapper(*args, t=None, timeperiod=None, **kwargs):
                """
                Wrapper that accepts both 't' and 'timeperiod' arguments
                for compatibility with existing code.
                """
                # Handle period argument (accept both t and timeperiod)
                period = t or timeperiod
                if period is not None:
                    # Pass as timeperiod to talib
                    kwargs['timeperiod'] = period
                    # Remove t if it exists in kwargs
                    kwargs.pop('t', None)
                
                return talib_func(*args, **kwargs)
            
            # Cache the wrapper as a static method for performance
            setattr(cls, name, staticmethod(smart_wrapper))
            return smart_wrapper
        
        raise AttributeError(
            f"'{cls.__name__}' has no attribute '{name}' and talib doesn't have it either"
        )


class Indicator(metaclass=IndicatorMeta):
    """
    Technical indicator library combining talib functions and custom implementations.
    
    - talib functions are automatically delegated (e.g., Indicator.RSI, Indicator.SMA)
    - Custom indicators are explicitly defined below
    - Both can be accessed through the same Indicator.XXX() interface
    """
    
    @classmethod
    def _getIndicators(cls):
        """Return list of all available indicator names."""
        # Get explicitly defined methods
        custom_indicators = [func for func in dir(cls) if not func.startswith("_")]
        # Add talib functions
        talib_indicators = [func for func in dir(talib) if not func.startswith("_") and callable(getattr(talib, func))]
        return list(set(custom_indicators + talib_indicators))
    
    @classmethod
    def _getSeriesIndicators(cls):
        """Return list of indicators marked for pandas Series registration."""
        return [
            getattr(cls, func_name) 
            for func_name, method in cls.__dict__.items() 
            if isinstance(method, staticmethod) and hasattr(method.__func__, '__series_indicator__')
        ]
    
    @classmethod
    def _getDataframeIndicators(cls):
        """Return list of indicators marked for pandas DataFrame registration."""
        return [
            getattr(cls, func_name) 
            for func_name, method in cls.__dict__.items() 
            if isinstance(method, staticmethod) and hasattr(method.__func__, '__dataframe_indicator__')
        ]
    
    # ========================================
    # CUSTOM INDICATORS (not in talib)
    # ========================================
    
    @staticmethod
    @seriesIndicator
    def PL(series):
        """Price Level: next period's price change."""
        return series.diff().shift(-1)
    
    @staticmethod
    def AO(high, low, t1=2, t2=3, ma_type=0):
        """Awesome Oscillator."""
        hl = (high + low) / 2
        ma_methods = [talib.SMA, talib.EMA, talib.WMA, talib.DEMA, talib.TEMA, talib.TRIMA, talib.KAMA, talib.T3]
        return (ma_methods[ma_type](hl, timeperiod=t1) - ma_methods[ma_type](hl, timeperiod=t2)).diff()
    
    @staticmethod
    @seriesIndicator
    def KELTNER_CHANNEL(high, low, close, t_ma=14, t_atr=14, multiplier=2.0):
        """Keltner Channel upper band."""
        ema = talib.EMA(close, timeperiod=t_ma)
        atr = talib.ATR(high, low, close, timeperiod=t_atr)
        return ema + multiplier * atr
    
    @staticmethod
    @seriesIndicator
    def BOLLINGERBANDS(close, t=5, std=2.0, ma_type=0):
        """Bollinger Bands - returns upper band only."""
        return talib.BBANDS(close, timeperiod=t, nbdevup=std, nbdevdn=std, matype=ma_type)[0]
    
    @staticmethod
    @seriesIndicator
    def MA(close, t=30, ma_type=0):
        """
        Universal Moving Average selector.
        Supports multiple MA types through a single interface.
        """
        ma_methods = [
            talib.SMA, talib.EMA, talib.WMA, talib.TSF, 
            talib.LINEARREG, talib.DEMA, talib.KAMA, talib.T3, 
            talib.TEMA, talib.TRIMA
        ]
        if t <= 1:
            return close
        else:
            try:
                return ma_methods[ma_type](close, timeperiod=t)
            except Exception as e:
                print(f"MA Error - ma_type: {ma_type}, t: {t}")
                raise e
    
    @staticmethod
    def PSYCHOLOGICALLINE(close, t=14):
        """Psychological Line indicator."""
        return close.mask(close.diff() < 0, 0).mask(close.diff() >= 0, 1).rolling(t).mean()
    
    @staticmethod
    @seriesIndicator
    def RCI(close, t=14):
        """
        Rank Correlation Index.
        Custom implementation not available in talib.
        """
        c = np.array(close.copy())
        rank_target = [np.roll(c, i, axis=-1) for i in range(t)]
        rank_target = np.vstack(rank_target)[:, t - 1:]
        price_rank = np.argsort(np.argsort(rank_target[::-1], axis=0), axis=0) + 1
        time_rank = np.arange(1, t + 1).reshape(t, -1)
        aa = np.sum((time_rank - price_rank)**2, axis=0, dtype=float) * 6
        bb = float(t * (t**2 - 1))
        cc = np.divide(aa, bb, out=np.zeros_like(aa), where=bb != 0)
        rci = (1 - cc) * 100
        rci = np.concatenate([np.full(t - 1, np.nan), rci], axis=0)
        return pd.Series(rci, index=close.index)
    
    @staticmethod
    @seriesIndicator
    def Volatility(close, t=1):
        """Log returns volatility."""
        volatility = np.log(close/close.shift(t))
        volatility.fillna(0, inplace=True)
        return volatility
    
    @staticmethod
    @seriesIndicator
    def HV(close, t=14):
        """Historical Volatility."""
        returns = np.log(close/close.shift(1))
        returns.fillna(0, inplace=True)
        volatility = returns.rolling(window=t).std() * np.sqrt(t)
        return volatility
    
    @staticmethod
    @dataframeIndicator
    def Heikinashi(dataframe, t_shift=0, t_ma=1, ma_type=0):
        """
        Heikin-Ashi candlestick transformation.
        Returns modified OHLC dataframe.
        """
        df = dataframe.copy()
        
        # Apply MA to OHLC if t_ma > 1
        if t_ma > 1:
            ma_methods = [
                talib.SMA, talib.EMA, talib.WMA, talib.TSF, 
                talib.LINEARREG, talib.DEMA, talib.KAMA, talib.T3
            ]
            ma_func = ma_methods[ma_type]
            
            df.Open = ma_func(df.Open, timeperiod=t_ma)
            df.High = ma_func(df.High, timeperiod=t_ma)
            df.Low = ma_func(df.Low, timeperiod=t_ma)
            df.Close = ma_func(df.Close, timeperiod=t_ma)
        
        o, h, l, c = df.Open.copy(), df.High.copy(), df.Low.copy(), df.Close.copy()
        df['Close'] = (o + h + l + c) / 4
        df['Open'] = (o + df['Close']) / 2
        
        idx = df.index.name
        df.reset_index(inplace=True)
        
        df['Open'] = Heikinashi_numba(df.Open.values, df.Close.values)
        
        if idx:
            df.set_index(idx, inplace=True)
        if t_shift > 0:
            df['Open'] = df['Open'].shift(t_shift)
        
        df['High'] = df[['Open','Close','High']].max(axis=1)
        df['Low'] = df[['Open','Close','Low']].min(axis=1)
        return df
    
    @staticmethod
    @dataframeIndicator
    def Zigzag(dataframe, percentage=0.05):
        """ZigZag indicator based on OHLC data."""
        df = dataframe.copy()
        high = df['High']
        low = df['Low']
        last_pivot_high = high[0]
        last_pivot_low = low[0]
        last_pivot_idx = 0
        zigzag = [np.nan] * len(df)
        current_trend = None
        
        for i in range(1, len(df)):
            if current_trend is None:
                if high[i] >= last_pivot_high * (1 + percentage):
                    current_trend = 1
                    last_pivot_high = high[i]
                    last_pivot_idx = i
                    zigzag[i] = last_pivot_high
                elif low[i] <= last_pivot_low * (1 - percentage):
                    current_trend = -1
                    last_pivot_low = low[i]
                    last_pivot_idx = i
                    zigzag[i] = last_pivot_low
            elif current_trend == 1:
                if high[i] >= last_pivot_high:
                    last_pivot_high = high[i]
                    zigzag[last_pivot_idx] = np.nan
                    zigzag[i] = last_pivot_high
                    last_pivot_idx = i
                elif low[i] <= last_pivot_high * (1 - percentage):
                    current_trend = -1
                    last_pivot_low = low[i]
                    zigzag[i] = last_pivot_low
                    last_pivot_idx = i
            elif current_trend == -1:
                if low[i] <= last_pivot_low:
                    last_pivot_low = low[i]
                    zigzag[last_pivot_idx] = np.nan
                    zigzag[i] = last_pivot_low
                    last_pivot_idx = i
                elif high[i] >= last_pivot_low * (1 + percentage):
                    current_trend = 1
                    last_pivot_high = high[i]
                    zigzag[i] = last_pivot_high
                    last_pivot_idx = i
            zigzag[-1] = df['Close'].iloc[-1]
        
        df['ZigZag'] = zigzag
        return df['ZigZag'].interpolate(method='linear')
    
    @staticmethod
    @seriesIndicator
    def Zigzag_Close(c, percentage=0.25):
        """ZigZag indicator for close prices only."""
        column = 'target'
        df = pd.DataFrame(index=c.index)
        df[column] = c.copy()
        df = df.dropna()
        zigzag = [np.nan] * len(df)
        last_extreme_index = 0
        last_extreme_value = df[column].iloc[0]
        direction = None  # "up" or "down"
        
        for i in range(1, len(df)):
            price = df[column].iloc[i]
            change = (price - last_extreme_value) / last_extreme_value
            
            if direction is None:  # Initial setup
                if abs(change) >= percentage:
                    direction = "up" if change > 0 else "down"
                    last_extreme_index = i
                    last_extreme_value = price
                    zigzag[i] = price
            elif direction == "up":
                if price > last_extreme_value:
                    last_extreme_index = i
                    last_extreme_value = price
                elif (last_extreme_value - price) / last_extreme_value >= percentage:
                    zigzag[last_extreme_index] = last_extreme_value
                    direction = "down"
                    last_extreme_index = i
                    last_extreme_value = price
                    zigzag[i] = price
            elif direction == "down":
                if price < last_extreme_value:
                    last_extreme_index = i
                    last_extreme_value = price
                elif (price - last_extreme_value) / last_extreme_value >= percentage:
                    zigzag[last_extreme_index] = last_extreme_value
                    direction = "up"
                    last_extreme_index = i
                    last_extreme_value = price
                    zigzag[i] = price
        
        zigzag[last_extreme_index] = last_extreme_value
        
        df["zigzag"] = zigzag
        df["zigzag"] = df["zigzag"].interpolate(method='linear')
        df.loc[(df["zigzag"].diff() < 0) & (df["zigzag"].diff().shift(-1) < 0), "zigzag"] = np.nan
        df.loc[(df["zigzag"].diff() > 0) & (df["zigzag"].diff().shift(-1) > 0), "zigzag"] = np.nan
        df["zigzag"] = df["zigzag"].interpolate(method='linear')
        df.loc[df["zigzag"]==0, "zigzag"] = np.nan
        df = df.reindex(c.index)
        return df["zigzag"].shift(-1)
    
    @staticmethod
    @dataframeIndicator
    def Ichimoku(dataframe, t1=9, t2=26, t3=52):
        """Calculate Ichimoku cloud components."""
        df = dataframe.copy()
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Conversion line (Tenkan-sen)
        nine_high = high.rolling(window=t1).max()
        nine_low = low.rolling(window=t1).min()
        df['tenkan_sen'] = (nine_high + nine_low) / 2
        
        # Base line (Kijun-sen)
        period26_high = high.rolling(window=t2).max()
        period26_low = low.rolling(window=t2).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Leading span 1 (Senkou Span A)
        df['senkou_span1'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(t2)
        
        # Leading span 2 (Senkou Span B)
        period52_high = high.rolling(window=t3).max()
        period52_low = low.rolling(window=t3).min()
        df['senkou_span2'] = ((period52_high + period52_low) / 2).shift(t2)
        
        # Lagging span (Chikou Span)
        df['chikou_span'] = close.shift(-t2)
        
        return df
    
    @staticmethod
    @dataframeIndicator
    def Ichimoku2(dataframe, t1=9, t2=26, t3=52):
        """Alternate Ichimoku cloud calculation."""
        df = dataframe.copy()
        
        # Base line (Kijun-sen)
        high26 = df.High.rolling(window=t2).max()
        low26 = df.Low.rolling(window=t2).min()
        df["base_line"] = (high26 + low26) / 2
        
        # Conversion line (Tenkan-sen)
        high9 = df.High.rolling(window=t1).max()
        low9 = df.Low.rolling(window=t1).min()
        df["conversion_line"] = (high9 + low9) / 2
        
        # Leading span 1 (Senkou Span A)
        leading_span1 = (df["base_line"] + df["conversion_line"]) / 2
        df["leading_span1"] = leading_span1.shift(t2)
        
        # Leading span 2 (Senkou Span B)
        high52 = df.High.rolling(window=t3).max()
        low52 = df.Low.rolling(window=t3).min()
        leading_span2 = (high52 + low52) / 2
        df["leading_span2"] = leading_span2.shift(t2)
        
        # Lagging span (Chikou Span)
        df["lagging_span"] = df.Close.shift(-t2)
        
        return df
    
    @staticmethod
    @seriesIndicator
    def KalmanMA(close: pd.Series, t=30):
        """Kalman Filter Moving Average."""
        ma = fast_kalman_filter(close.values, 0.01, 0.01*t)
        return pd.Series(ma, index=close.index)
    
    @staticmethod
    @seriesIndicator
    def GarmanKlassVolatility(_open, high, low, close, t=20):
        """Garman-Klass volatility estimator."""
        log_hl = np.log(high / low)
        log_co = np.log(close / _open)
        gk_vol_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return gk_vol_daily.rolling(window=t).mean()
    
    # Note: SUPERTREND is commented out in original - can be uncommented if needed
    # @staticmethod
    # def SUPERTREND(high, low, close, t=14, factor=2.0, only_result=False):
    #     """SuperTrend indicator (requires pandas_ta)."""
    #     import pandas_ta as pta
    #     result = pta.supertrend(high, low, close, length=t, multiplier=factor)
    #     return result.iloc[:, 1] if only_result else result


# ========================================
# NUMBA-OPTIMIZED HELPER FUNCTIONS
# ========================================

@njit(cache=True)
def fast_kalman_filter(close: np.ndarray, Q: float, R: float) -> np.ndarray:
    """Compute a fast Kalman filter for a 1D close price series."""
    n = len(close)
    estimates = np.zeros(n)
    
    # Initialize variables
    estimates[0] = close[0]
    p_k = 1.0  # Initial covariance of errors
    
    # Iterate over remaining data points
    for k in range(1, n):
        # Prediction step
        p_pred = p_k + Q
        
        # Update step
        kalman_gain = p_pred / (p_pred + R)
        estimates[k] = estimates[k-1] + kalman_gain * (close[k] - estimates[k-1])
        p_k = (1 - kalman_gain) * p_pred
    
    return estimates

@njit(cache=True)
def Heikinashi_numba(Open, Close):
    """Vectorized helper to compute Heikin-Ashi open prices."""
    for i in range(1, len(Open)):  # Start from index 1
        if 1 != 0 and not np.isnan(Open[i-1]) and not np.isnan(Close[i-1]):
            Open[i] = (Open[i-1] + Close[i-1]) / 2
    return Open
