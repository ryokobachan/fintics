import logging
from typing import Literal, Optional
import numpy as np
import pandas as pd
from fintics.strategy import Strategy, OptimizeStrategy
from fintics.strategy.orderprice import Marker_OrderPrice
from fintics.backtest import Backtest
from fintics.backtest.util import BacktestResult, BacktestPriceTypeList, optimize

class RealtimeBacktest(Backtest):
    def __init__(self, df_tick: pd.DataFrame, timeframe: None|str=None, **kwargs):
        _df_tick = self._init_df_tick_raw(df_tick.copy())
        _df = self._init_df_tick(_df_tick.copy())
        if timeframe is not None:
            _df = self._init_df(_df, timeframe)

        self._df = _df
        self._timeframe = timeframe
        self.logger = logging.getLogger(__name__)

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _init_df_tick_raw(self, df_tick, reverse_side_status:bool=True):
        if reverse_side_status:
            df_tick['side'] = df_tick['side'].replace({'buy': 'sell', 'sell': 'buy'})

        return df_tick.assign(
            buy_available = (
                df_tick['price']
                .mask(df_tick['side'] != 'buy', np.nan)
                .bfill()
                .mask(lambda x: x.diff().shift(-1) >= 0, np.nan)
            ),
            sell_available = (
                df_tick['price']
                .mask(df_tick['side'] != 'sell', np.nan)
                .bfill()
                .mask(lambda x: x.diff().shift(-1) <= 0, np.nan)
            )
        )
    
    def _init_df_tick(self, df_tick):
        return (
            df_tick.assign(
                volume_buy = df_tick['amount'].mask(df_tick['side'] == 'sell', 0),
                volume_sell = df_tick['amount'].mask(df_tick['side'] == 'buy', 0),
                executed_at = df_tick.index,
                _price = df_tick['price'] - (df_tick['side']=='sell') * 1,
                _price_group = lambda x: x['_price'].diff().abs().STEP().cumsum()
            )
            .groupby('_price_group')
            .agg(
                Price=('_price', 'first'),
                _BuyAvailablePrice=('buy_available', 'last'), #future data
                _SellAvailablePrice=('sell_available', 'last'), #future data
                Volume_buy=('volume_buy', 'sum'),
                Volume_sell=('volume_sell', 'sum'),
                index=('executed_at', 'first'),
            )
            .set_index('index', drop=True)
            .dropna(subset=['_BuyAvailablePrice', '_SellAvailablePrice'], how='all')
        )

    def _init_df(self, df_tick, timeframe: str):
        return(
            df_tick['Price']
            .resample(timeframe)
            .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
            .assign(
                # buy
                Open_buy = df_tick['_BuyAvailablePrice'].resample(timeframe).first(),
                High_buy = df_tick['_BuyAvailablePrice'].resample(timeframe).max(),
                Low_buy = df_tick['_BuyAvailablePrice'].resample(timeframe).min(),
                Close_buy =  df_tick['_BuyAvailablePrice'].resample(timeframe).last(),
                Volume_buy = df_tick['Volume_buy'].resample(timeframe).sum(),
                _BuyAvailablePrice =  df_tick['_BuyAvailablePrice'].resample(timeframe).min().shift(-1), #future data
                # sell
                Open_sell = df_tick['_SellAvailablePrice'].resample(timeframe).first(),
                High_sell = df_tick['_SellAvailablePrice'].resample(timeframe).max(),
                Low_sell = df_tick['_SellAvailablePrice'].resample(timeframe).min(),
                Close_sell =  df_tick['_SellAvailablePrice'].resample(timeframe).last(),
                Volume_sell = df_tick['Volume_sell'].resample(timeframe).sum(),
                _SellAvailablePrice =  df_tick['_SellAvailablePrice'].resample(timeframe).max().shift(-1), #future data

                Volume = lambda _d: _d['Volume_buy'] + _d['Volume_sell']
            )
            .dropna(subset=['Close'])
        )

    def run(
            self, 
            strategy: Strategy|None = None, 
            params: dict={}, 
            start: Optional[str] = None, 
            end: Optional[str] = None, 
            orderprice = Marker_OrderPrice, 
            fee_rate = -0.0002, 
            spread = -1.000, 
            only_buy: bool = True, 
            reinvest: bool = False, 
            reverse: bool = False, 
            price_type: BacktestPriceTypeList = 'Open'
            ):
        _df = self._get_df(start, end)

        def assign_orderprice(_df):
            _buy, _sell = orderprice(_df.copy())
            return _df.assign(BuyOrderPrice=_buy, SellOrderPrice=_sell)
        
        def runwise_best_multi(df, key_col, specs):
            y = df[key_col].to_numpy()
            n = len(df)
            if n == 0:
                return {new: df[src].copy() for new,(src,_) in specs.items()}

            idx = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
            counts = np.diff(np.r_[idx, n])

            out = {}
            for new, (src, how) in specs.items():
                a = df[src].to_numpy(dtype='float64', copy=True)
                nan = np.isnan(a)
                if how == 'min':
                    a[nan] = np.inf
                    red = np.minimum.reduceat(a, idx)
                elif how == 'max':
                    a[nan] = -np.inf
                    red = np.maximum.reduceat(a, idx)
                else:
                    raise ValueError("how must be 'min' or 'max'")
                valid = np.add.reduceat((~nan).astype(np.int64), idx)
                red[valid == 0] = np.nan
                out[new] = pd.Series(np.repeat(red, counts), index=df.index)

            return out
        
        _df = (
            _df
            .dropna(subset=['Open', 'High', 'Low', 'Close'])
            .assign(
                Price = lambda d: d['Close'] if price_type == 'Close' else d['Open'].shift(-1),
                _y = lambda x: strategy(x, **params).STEP() if strategy is not None else (0 if 'y' not in x.columns else x['y']),
                _signal = lambda x: x['_y'].diff().fillna(0),
                _order_status = lambda x: x['_signal'].mask(lambda s: s==0, np.nan).ffill().fillna(0),
            )
            .pipe(lambda x: x.assign(**runwise_best_multi(x, '_y', {
                '_BuyOrderBestPrice':  ('_BuyAvailablePrice',  'min'),
                '_SellOrderBestPrice': ('_SellAvailablePrice', 'max'),
            })))
            .pipe(assign_orderprice)
            .assign(
                BuySettleablePrice = lambda x: x['BuyOrderPrice'].mask((x['BuyOrderPrice'] < x['_BuyAvailablePrice']) | (x['_BuyAvailablePrice'].isna()), np.nan),
                SellSettleablePrice = lambda x: x['SellOrderPrice'].mask((x['SellOrderPrice'] > x['_SellAvailablePrice']) | (x['_SellAvailablePrice'].isna()), np.nan),
                y = lambda d: (
                    d['_y']
                    .mask((d['_order_status']>0) & (d['BuySettleablePrice'].isna()), np.nan)
                    .mask((d['_order_status']<0) & (d['SellSettleablePrice'].isna()), np.nan)
                    .ffill()
                ),
                signal = lambda x: x['y'].diff().fillna(0),
                _price = lambda x: x['Price'].mask(x['signal'] > 0, x['BuySettleablePrice']).mask(x['signal'] < 0, x['SellSettleablePrice']),
                Spread = spread,
                Fee = lambda x: x['_price'] * fee_rate
            )
        )

        self._backtest_df = BacktestResult(
            _df,
            strategy=strategy,
            params=params,
            only_buy=only_buy,
            reinvest=reinvest,
            reverse=reverse,
        )

        return self.get_performance()
    
    def optimize(
        self,
        strategy=None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        orderprice = None, 
        only_buy: Optional[bool] = True,
        reverse: bool = False,
        reinvest: bool = False,
        price_type: Literal['Close', 'Open'] = 'Open',
        fee_rate: Optional[float] = -0.0002,
        spread: Optional[float] = -1.0,
        n_trials: int = 100,
        target: Literal['profit', 'win_rate', 'max_drawdown', 'max_drawdown_rate', 'profitfactor', 'sharperatio', 'sqn'] = 'profit',
        max_results: Optional[int] = 50,
    ):
        """Optimize a strategy's parameters using Optuna."""

        _performances = []

        @optimize(n_trials, description=strategy.__name__, tqdm_message=lambda study, trial: f'best score: {study.best_value}, params: {study.best_trial.params}')
        def study(trial):
            self.run(
                strategy=strategy,
                params={'trial': trial},
                start=start,
                end=end,
                orderprice = lambda _df: orderprice(_df, trial=trial),
                only_buy=only_buy,
                reinvest=reinvest,
                reverse=reverse,
                price_type=price_type,
                spread=spread,
                fee_rate=fee_rate,
            )
            _performance = self.get_performance()
            _performances.append(_performance)
            return _performance[target]

        self._performance = (
            pd.DataFrame(_performances)
            .set_index('strategy', drop=True)
            .drop_duplicates('params')
            .sort_values(target, ascending=False)
            .head(max_results)
        )

        self.run(
            strategy=strategy,
            params={'trial': study.study.best_trial},
            start=start,
            end=end,
            orderprice = lambda _df: orderprice(_df, trial=study.study.best_trial),
            only_buy=only_buy,
            reinvest=reinvest,
            reverse=reverse,
            price_type=price_type,
            spread=spread,
            fee_rate=fee_rate,
        )

        return self._performance

    def optimize_all_strategy(
        self,
        strategies: Optional[list] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        orderprice = None, 
        fee_rate: Optional[float] = -0.0002,
        spread: Optional[float] = -1.0,
        only_buy: Optional[bool] = True,
        reverse: bool = False,
        reinvest: bool = False,
        price_type: Literal['Close', 'Open'] = 'Open',
        n_trials: int = 100,
        target: Literal['profit', 'win_rate', 'max_drawdown', 'max_drawdown_rate', 'profitfactor', 'sharperatio', 'sqn'] = 'profit',
    ):
        """Optimize parameters across multiple strategies."""
        
        if strategies is None:
            strategies = OptimizeStrategy.__subclasses__()
        
        _all_performances = []
        for strategy in strategies:
            _performance = self.optimize(
                strategy=strategy,
                start=start,
                end=end,
                orderprice = orderprice,
                only_buy=only_buy,
                reinvest=reinvest,
                reverse=reverse,
                price_type=price_type,
                n_trials=n_trials,
                target=target,
                spread=spread,
                fee_rate=fee_rate
            )
            _all_performances.append(_performance.reset_index().iloc[0].to_dict())

        # show the results
        self._performance = (
            pd.DataFrame(_all_performances)
            .set_index('strategy', drop=True)
            .sort_values(target, ascending=False)
        )

        return self._performance
