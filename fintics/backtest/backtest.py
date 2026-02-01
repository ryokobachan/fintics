"""High level backtesting interface for evaluating strategies."""

import logging
from typing import Optional
import pandas as pd
from fintics.backtest.optimizer import BacktestOptimize
from fintics.backtest.plot import BacktestPlot
from fintics.backtest.util import BacktestedDataFrame, BacktestPriceTypeList
from fintics.strategy import Strategy


class Backtest(BacktestOptimize, BacktestPlot):
    """Run backtests on OHLCV data and retrieve results."""

    df_columns = ['Open', 'High', 'Low', 'Close']

    def __init__(self, df: pd.DataFrame):
        self._df = df

        # check dataframe
        missing = [col for col in self.df_columns if col not in df.columns]
        if missing:
            raise Exception(f"Dataframe requires: {missing}")

        self.logger = logging.getLogger(__name__)

    def _set_logging_level(self, verbose: bool) -> None:
        """Adjust logger verbosity."""

        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _get_df(self, start=None, end=None):
        """Return dataframe sliced between ``start`` and ``end``."""

        return self._df.Range(start, end).copy()

    def _get_backtest_df(self):
        """Return dataframe used for last backtest run."""

        assert hasattr(self, '_backtest_df'), 'Required to run backtest.'
        return self._backtest_df

    def run(
        self,
        strategy: Optional[Strategy] = None,
        params: dict = {},
        start: Optional[str] = None,
        end: Optional[str] = None,
        leverage: float = 1.0,
        spread: Optional[float] = 0,
        fee_rate: Optional[float] = 0.0,
        only_buy: bool = True,
        reverse: bool = False,
        price_type: BacktestPriceTypeList = 'Open',
    ) -> None:
        """Execute backtest with given strategy and parameters."""

        _df = self._get_df(start, end)

        self._backtest_df = BacktestedDataFrame(
            _df,
            strategy=strategy,
            params=params,
            leverage=leverage,
            only_buy=only_buy,
            price_type=price_type,
            reverse=reverse,
            spread=spread,
            fee_rate=fee_rate,
        )

        return self.get_performance()

    def get_trades(self):
        """Return dataframe of executed trades."""

        df = self._get_backtest_df()
        return df.trades

    def get_performance(self):
        """Return performance summary of last backtest."""

        df = self._get_backtest_df()
        return df.getInfo()
