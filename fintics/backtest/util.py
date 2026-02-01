"""Utility extensions and helpers used by the backtesting module."""

from typing import Literal, Optional, Union
import json
import numpy as np
import pandas as pd
import pandas_flavor as pf
import optuna
from tqdm import tqdm
import functools
# Remove circular import - import Strategy only when needed

def optimize(n_trials, direction="maximize", show_progress_bar=True, description: str|None=None, tqdm_message=lambda study, trial: f'best score: {study.best_value}', **study_kwargs):
    """
    A decorator factory that runs Optuna optimization and stores 
    the result in the function's .study attribute.
    Includes an optional tqdm progress bar.
    
    Args:
        n_trials (int): The number of trials.
        direction (str): 'minimize' or 'maximize'.
        show_progress_bar (bool): Whether to display the tqdm progress bar.
        **study_kwargs: Other arguments to pass to optuna.create_study()
                          (e.g., sampler=optuna.samplers.TPESampler())
    """
    
    def decorator(objective_func):
        """
        The actual decorator function for the objective.
        """
        
        @functools.wraps(objective_func)
        def wrapper_with_study(*args, **kwargs):
            # Typically, the function wrapped by this decorator (objective)
            # is not expected to be called directly.
            # If it is, just execute the original function.
            return objective_func(*args, **kwargs)

        # --- Decorator's main logic (executes when the function is defined) ---
        
        # 1. Create the Study
        study = optuna.create_study(direction=direction, **study_kwargs)
        
        # 2. Prepare for optimization (with or without tqdm)
        
        callbacks_list = []
        pbar = None # Initialize pbar to None

        if show_progress_bar:
            # Initialize tqdm
            pbar = tqdm(total=n_trials, desc=description)
            
            def tqdm_callback(study, trial):
                """Updates the tqdm progress bar after each trial."""
                pbar.set_postfix_str(tqdm_message(study, trial))
                pbar.update(1)
            
            callbacks_list.append(tqdm_callback)

        # 3. Run the optimization
        try:
            study.optimize(
                objective_func, 
                n_trials=n_trials,
                # Use the callbacks list if it's not empty, otherwise None
                callbacks=callbacks_list if callbacks_list else None 
            )
        finally:
            # Ensure the progress bar is closed if it was created
            if pbar:
                pbar.close()

        # 4. Attach the result (study) as an attribute
        wrapper_with_study.study = study
        
        # ---
        
        # 5. Return the wrapper function
        return wrapper_with_study

    return decorator

# Register pandas extensions immediately when util is imported
@pf.register_dataframe_method
def get_row(self, index: int):
    """Return row at ``index`` or NaNs if out of range."""

    if index > -1 and index < len(self):
        return self.iloc[index]
    else:
        return pd.DataFrame(np.zeros((1, len(self.columns))), columns=self.columns).replace(0, np.nan).iloc[0]

@pf.register_dataframe_method
def ohlc(self, method='D'):
    """Resample to OHLCV using the given method."""

    return self.resample(method).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

@pf.register_dataframe_method
def Range(self, start=None, end=None):
    """Filter dataframe between ``start`` and ``end`` dates."""
    if start is None:
        start = '1900-01-01'
    if end is None:
        end = '2099-12-31'
    return self.loc[start:end]

@pf.register_dataframe_method
def update_df(self, new_df):
    """Combine current dataframe with ``new_df`` and sort by index."""

    df = new_df.combine_first(self).sort_index()
    return df

@pf.register_dataframe_method
def remove_no_trade(df):
    """Remove rows where trading volume is zero."""

    return df.loc[df['Volume'] > 0]

@pf.register_dataframe_method
def X(self):
    """Return dataframe filtered to columns starting with ``X``."""

    return self.filter(like='X')

@pf.register_dataframe_method
def X_OHLCV(df):
    """Return OHLCV columns along with ``X_`` feature columns."""

    ohlcv_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    x_features_df = df.filter(like='X')
    combined_df = pd.concat([ohlcv_df, x_features_df], axis=1)
    return combined_df

@pf.register_dataframe_method
def X_columns(self):
    """Return list of column names starting with ``X_``."""

    return [col for col in self.columns if col.startswith('X_')]

@pf.register_series_method
def zerona(self):
    """Replace zero values with NaN."""

    return self.mask(self == 0, np.nan)

@pf.register_series_method
def seq(series, seq_type=0):
    """Return sequence counts for grouped values."""

    '''
    seq_type=0: [0,0,1,1,1,-1,0] -> [2,2,3,3,3,1,1]
    seq_type=1: [0,0,1,1,1,-1,0] -> [1,2,1,2,3,1,1]
    '''
    if seq_type == 0:
        # Identify where the value changes
        change_points = series != series.shift()

        # Group consecutive elements and count the size of each group
        counts = change_points.cumsum().map(series.groupby(change_points.cumsum()).size())
        output = counts.tolist()

        return pd.Series(output, index=series.index)
    elif seq_type == 1:
        return series.groupby((series != series.shift()).cumsum()).cumcount() + 1

@pf.register_series_method
def SEQ(series, grad=4):
    """Scale sequence length and multiply by original series."""

    s = series.seq(seq_type=1)
    s.loc[s > grad] = grad
    s /= grad
    return s * series

@pf.register_series_method
def STEP(y, signal=0):
    """Convert values to binary step output relative to ``signal``."""

    return y.mask(y > signal, 1).mask(y < signal, 0).fillna(0)

@pf.register_series_method
def STEP_MULTI(y, signal=0):
    """Return -1, 0, or 1 depending on ``signal`` threshold."""

    return y.mask(y > signal, 1).mask(y < -signal, -1).mask((-signal <= y) & (y <= signal), 0)

@pf.register_dataframe_method
def y_timeframe(df, strategy, params, timeframe, origin_timeframe, reverse: bool = False):
    """Calculate signals on different timeframe and map back to origin."""

    df_calc = df.ohlc(timeframe)
    df_calc['y'] = strategy(df_calc, **params) * (-1 if reverse else 1)
    df_calc['y'] = df_calc['y'].shift()
    return df_calc['y'].resample(origin_timeframe).ffill()

def _required_round(func, decimals=4):
    """Return wrapper rounding result to ``decimals`` places."""

    return lambda *args, **kwargs: round(func(*args, **kwargs), decimals)

from scipy.stats import chatterjeexi

@pf.register_dataframe_method
def corr_xi(self):
    return self.corr(method=lambda x, y: chatterjeexi(x, y).statistic)

@pf.register_dataframe_method
def train_test_split(df, test_size=0.2):
    n = len(df)
    split_index = int(n * (1 - test_size))
    X = df.X()
    y = df['y']
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

BacktestPriceTypeList = Literal['Close', 'Open']

class BacktestPriceType():
    @classmethod
    def Close(self, df) -> pd.Series:
        """Return close price series."""
        return df.Close

    @classmethod
    def Open(self, df) -> pd.Series:
        """Return next period open price series."""
        return df.Open.shift(-1)

class BacktestedDataFrame(pd.DataFrame):
    """Dataframe containing results and metadata for a backtest run."""

    def __init__(self, df: pd.DataFrame, strategy=None, params: Optional[dict] = None, leverage: float = 1.0, only_buy: bool = True, price_type: BacktestPriceTypeList = 'Close', reverse: bool = False, spread: float = 0.000, fee_rate: float = 0, deposit=100):
        params = params or {}

        if 'y' in df.columns:
            if reverse:
                df['y'] = df['y'] * (-1)
            if only_buy:
                df['y'].loc[df['y'] < 0] = 0
        elif strategy is not None:
            df['y'] = strategy(df, **params).getY(only_buy=only_buy, reverse=reverse)
        else:
            df['y'] = 0
        
        df['y'] = df['y'] * leverage
        
        df['y'] = df.get('y', 0).fillna(0)
        df.loc[df.index[0], 'y'] = 0
        df['signal'] = df['y'].diff().fillna(0)
        df['Spread'] = df.get('Spread', spread)
        df['_price'] = df.get('_price', getattr(BacktestPriceType, price_type)(df))    
        df['Fee'] = df.get('Fee', df['_price'] * fee_rate)    

        _price = df['_price']
        _price_changerate = _price.pct_change(fill_method=None).shift(-1).fillna(0)
        _price_adjusted = df['_price'] + df['signal'] * (df['Spread'] + df['Fee'])
        _price_changerate_adjusted = _price_adjusted.pct_change(fill_method=None).shift(-1).fillna(0)

        df['pl_hist'] = deposit * (1 + df['y'] * _price_changerate_adjusted).cumprod()
        df['pl'] = df['pl_hist'].diff().fillna(0)

        df['pl_growth'] = 1
        df.loc[df.index[0], 'pl_growth'] = 0
        df['pl_growth'] = deposit * (1 + df['pl_growth'] * _price_changerate).cumprod()


        super().__init__(df)

        # Use proper attribute assignment instead of direct assignment
        object.__setattr__(self, '_strategy', strategy)
        object.__setattr__(self, '_params', params)
        object.__setattr__(self, '_only_buy', only_buy)
        object.__setattr__(self, '_deposit', deposit)
        try:
            if 'trial' in params.keys():
                _trial = params.pop('trial')
                _params = json.dumps({**_trial.params, **params})
            else:
                _params = json.dumps(params)
        except:
            _params = {}
        
        self.__init_trades__()
        # Use proper attribute assignment
        object.__setattr__(self, '_info', {
            'strategy': strategy.__name__ if strategy is not None and getattr(strategy, '__name__') else 'Unknown Strategy',
            'params': _params,
            'start': df.index[0].strftime('%Y-%m-%d %H:%M'),
            'end': df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'only_buy': bool(only_buy),
            'price_type': price_type
        })

    def __init_trades__(df):
        _df = df.copy()
        _df['_t'] = _df.index
        _df['_t'] = _df['_t'].shift(-1)

        trades = _df.loc[_df['signal'] != 0]

        trades['entry'] = trades['_t']
        trades['exit'] = trades['entry'].shift(-1)

        trades = trades[['entry', 'exit', 'pl', 'pl_hist', 'pl_growth', 'signal']]
        trades['pl_hist'] = trades['pl_hist'].shift(-1)
        trades['pl_growth'] = trades['pl_growth'].shift(-1)
        trades = trades.loc[trades['signal'] > 0]

        trades['_pl_hist_prev'] = trades['pl_hist'].shift().fillna(df._deposit)
        trades['pl'] = trades['pl_hist'] - trades['_pl_hist_prev']

        trades = trades.drop(['signal', '_pl_hist_prev'], axis=1)

        trades = trades.reset_index(drop=True)
        trades.index += 1

        object.__setattr__(df, 'trades', trades)
    
    def _required_trades(func):
        def inner(self, *args, **kwargs):
            if not hasattr(self, 'trades'):
                self.__init_trades__()
            return func(self, *args, **kwargs)
        return inner
        
    def getInfo(df):
        return BacktestResult(df)
 
    def Pl(df):
        return df['pl']
    
    def PlHist(df):
        return df['pl_hist']
    
    def PlHistSimple(df):
        return df[f'pl_hist_simple']
    
    def Drawdown(df):
        return df.PlHist() - df.PlHist().cummax()
    
    def Drawdown_Growth(df):
        return df['pl_growth'] - df['pl_growth'].cummax()
    
    def PlGrowth(df):
        return df[f'pl_growth']
    
    def EfficiencyRatio(df):
        ideal_trade_duration = len(df.loc[df['_price'].diff() > 0])
        trade_duration = len(df.loc[(df['_price'].diff() > 0) & (df['y'] > 0)])
        if ideal_trade_duration == 0:
            return 0
        return trade_duration / ideal_trade_duration
    
    @_required_round
    def PD(df):
        profit = df.Profit()
        mdd = -df.MaxDrawdownRate()
        if profit <= 0 or mdd == 0:
            return -np.inf
        return profit / mdd
    
    @_required_round
    def PDE(df):
        profit = df.Profit()
        mdd = -df.MaxDrawdownRate()
        effi = df.EfficiencyRatio()
        if profit <= 0 or mdd == 0:
            return -np.inf
        return profit / mdd * effi
    
    @_required_round
    def PD_G(df):
        profit = df.Profit()
        mdd = -df.MaxDrawdownRate()
        if profit <= 0 or mdd == 0:
            return -np.inf
        return (profit / df.Growth()) / (df.MaxDrawdownRate() / df.MaxDrawdownRate_Growth())
    
    @_required_round
    def PDE_G(df):
        profit = df.Profit()
        mdd = -df.MaxDrawdownRate()
        if profit <= 0 or mdd == 0:
            return -np.inf
        return (profit / df.Growth()) / (df.MaxDrawdownRate() / df.MaxDrawdownRate_Growth()) * df.EfficiencyRatio()
    
    @_required_round
    def MaxDrawdown(df):
        return df.Drawdown().min()
    
    @_required_round
    def MaxDrawdownRate(df):
        profit = df.Profit()
        if profit <= 0:
            return -np.inf
        return df.MaxDrawdown() / df.Profit()
    
    @_required_round
    def MaxDrawdown_Growth(df):
        return df.Drawdown_Growth().min()
    
    @_required_round
    def MaxDrawdownRate_Growth(df):
        profit = df.Growth()
        if profit <= 0:
            return -np.inf
        return df.MaxDrawdown_Growth() / df.Growth()
    
    def DrawdownDuration(df):
        return str(df.PlHist().cummax().seq(seq_type=1).max() * (df.index[1] - df.index[0]))
    
    @_required_round
    def DrawdownAreaAVG(df):
        return df.Drawdown().sum() / len(df)
    
    @_required_round
    def Profit(df):
        profit = df['pl_hist'].iloc[-1]
        return profit
    
    @_required_round
    def ProfitBase(df):
        profit = (df['y'] * df['_price'].diff().shift(-1)).sum()
        return profit
    
    @_required_round
    def Growth(df):
        growth = df['pl_growth'].iloc[-1]
        return growth

    @_required_round
    def SharpeRatio(df, risk_free_rate=0):
        closes = df.loc[df['y'] != 0, 'Close']
        if len(closes) == 0:
            return 0
        returns = df.PlHist().pct_change(fill_method=None)
        returns = returns.loc[(returns != np.inf) & (returns != -np.inf)]
        sharperatio = (returns.mean() - risk_free_rate) * np.sqrt(len(returns)) / returns.std()
        if np.isnan(sharperatio):
            return 0
        return sharperatio
    
    @_required_round
    def SortinoRatio(df, risk_free_rate=0):
        closes = df.loc[df['y'] != 0, 'Close']
        if len(closes) == 0:
            return 0

        returns = df.PlHist().pct_change(fill_method=None)
        returns = returns.loc[(returns != np.inf) & (returns != -np.inf)]
        downside_risk = np.sqrt((returns[returns < 0] ** 2).mean())
        sortino_ratio = ((returns.mean() - risk_free_rate) * np.sqrt(len(returns)) / downside_risk) if downside_risk != 0 else np.inf
        return sortino_ratio
    
    @_required_trades
    def TradesNum(df):
        trades = df.trades
        n_trades = len(trades)
        return n_trades
    
    @_required_round
    @_required_trades
    def ExposureDurationRate(df):
        exposure_duration_rate = df['y'].mean()
        return exposure_duration_rate
    
    @_required_round
    @_required_trades
    def Profit_divided_BestProfit(df):
        pl = df['_price'].diff().shift(-1)
        if df._only_buy:
            pl_sum = pl.loc[pl>0].sum()
        else:
            pl_sum = pl.abs().sum()
        # Avoid division by zero
        if pl_sum == 0 or np.isnan(pl_sum):
            return 0
        return df.ProfitBase() / pl_sum
    
    @_required_round
    @_required_trades
    def WinRate(df):
        trades = df.trades
        win_rate = (trades.pl > 0).mean() * 1
        if np.isnan(win_rate):
            return 0
        else:
            return win_rate
    
    @_required_trades
    @_required_round
    def SQN(df):
        trades = df.trades
        sqn = np.sqrt(len(trades)) * trades.pl.mean() / (trades.pl.std() or np.nan)
        if np.isnan(sqn):
            sqn = 0
        return sqn
    
    @_required_trades
    @_required_round
    def ProfitFactor(df):
        trades = df.trades
        positive_pl = trades.loc[trades['pl']>0, 'pl'].sum()
        negative_pl = trades.loc[trades['pl']<0, 'pl'].sum()
        # Avoid division by zero
        if negative_pl == 0 or np.isnan(negative_pl):
            return np.inf if positive_pl > 0 else 0
        return positive_pl / negative_pl * (-1)
    
    @_required_round
    @_required_trades
    def AverageProfit(df):
        trades = df.trades
        if len(trades) == 0:
            return 0
        return trades['pl'].mean()

class BacktestResult(pd.Series):
    def __init__(self, df: BacktestedDataFrame):
        super().__init__(pd.DataFrame({
            **df._info,
            'profit': df.Profit(),
            'growth': df.Growth(),
            'avg_profit': df.AverageProfit(),
            'win_rate': df.WinRate(),
            'n_trades': df.TradesNum(),
            'efficiency': df.ExposureDurationRate(),
            'profit/best_profit': df.Profit_divided_BestProfit(),
            'PD': df.PD(),
            'PD_vs_Growth': df.PD_G(),
            'PDE': df.PDE(),
            'PDE_vs_Growth': df.PDE_G(),
            'max_dd': df.MaxDrawdown(),
            'max_dd_rate': df.MaxDrawdownRate(),
            'max_dd_duration': df.DrawdownDuration(),
            'dd_avg': df.DrawdownAreaAVG(),
            'sqn': df.SQN(),
            'profitfactor': df.ProfitFactor(),
            'sharperatio': df.SharpeRatio(),
            'sortinoratio': df.SortinoRatio()
        }, index=[0]).loc[0])
    
    def summary(self):
        return self[[
            'profit',
            'growth',
            'win_rate',
            'n_trades',
            'efficiency',
            'PD',
            'PDE',
            'max_dd_rate',
            'max_dd_duration',
            'sqn',
            'profitfactor',
            'sharperatio'
        ]]
        