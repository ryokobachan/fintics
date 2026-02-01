"""Optimization utilities for tuning strategy parameters."""

from typing import Literal, Optional
import pandas as pd
import optuna
optuna.logging.disable_default_handler()
from tqdm import tqdm
from fintics.strategy import Strategy, OptimizeStrategy
from fintics.backtest.util import optimize


class BacktestOptimize:
    """Mixin class providing optimization routines."""

    def optimize(
        self,
        strategy=None,
        params: dict = {},
        start: Optional[str] = None,
        end: Optional[str] = None,
        leverage: float = 1.0,
        only_buy: Optional[bool] = True,
        reverse: bool = False,
        price_type: Literal['Close', 'Open'] = 'Open',
        spread: Optional[float] = 0,
        fee_rate: Optional[float] = 0.0,
        n_trials: int = 100,
        target: Literal['profit', 'win_rate', 'max_drawdown', 'max_drawdown_rate', 'profitfactor', 'sharperatio', 'sqn'] = 'profit',
        max_results: Optional[int] = 30,
    ):
        """Optimize a strategy's parameters using Optuna."""

        performances = []

        # objective function
        def tqdm_message(study, trial):
            p = {**study.best_trial.params, **params}
            return f'[{strategy.__name__}] best score: {round(study.best_value, 4)}, params: {p}'
        
        @optimize(n_trials, tqdm_message=tqdm_message)
        def study(trial):
            self.run(
                strategy=strategy,
                params={'trial': trial, **params},
                start=start,
                end=end,
                leverage=leverage,
                only_buy=only_buy,
                reverse=reverse,
                price_type=price_type,
                spread=spread,
                fee_rate=fee_rate,
            )
            performance = self.get_performance()
            performances.append(performance)
            return performance[target]

        self._optimize_history = (
            pd.DataFrame(performances)
            .reset_index(drop=True)
            .drop_duplicates('params')
            .sort_values('profit', ascending=False)
        )

        self.run(
            strategy=strategy,
            params={**study.study.best_params, **params},
            start=start,
            end=end,
            leverage=leverage,
            only_buy=only_buy,
            reverse=reverse,
            price_type=price_type,
            spread=spread,
            fee_rate=fee_rate,
        )

        # show the results
        if max_results is not None:
            self._optimize_history = self._optimize_history.head(max_results)

        return self._optimize_history

    def optimize_all_strategy(
        self,
        strategies: Optional[list] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        leverage: float = 1.0,
        only_buy: Optional[bool] = True,
        reverse: bool = False,
        price_type: Literal['Close', 'Open'] = 'Open',
        spread: Optional[float] = 0.0,
        fee_rate: Optional[float] = 0.0,
        n_trials: int = 100,
        target: Literal['profit', 'win_rate', 'max_drawdown', 'max_drawdown_rate', 'profitfactor', 'sharperatio', 'sqn'] = 'profit',
    ):
        """Optimize parameters across multiple strategies."""

        if strategies is None:
            strategies = OptimizeStrategy.__subclasses__()

        results = {}
        for strategy in strategies:
            self.optimize(
                start=start,
                end=end,
                leverage=leverage,
                strategy=strategy,
                only_buy=only_buy,
                reverse=reverse,
                price_type=price_type,
                n_trials=n_trials,
                target=target,
                spread=spread,
                fee_rate=fee_rate
            )
            results[strategy.__name__] = self.get_performance()

        # show the results
        self._optimize_all_strategy_history = pd.DataFrame(results).T.sort_values(target, ascending=False)

        # if only_outperformed_result:
        #     self._optimize_all_strategy_history = self._optimize_all_strategy_history.loc[self._optimize_all_strategy_history['profit'] > self._optimize_all_strategy_history['growth']]

        return self._optimize_all_strategy_history

