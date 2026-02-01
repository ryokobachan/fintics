"""LightGBM based classifier with custom evaluation metrics."""

from typing import Optional
from tqdm import tqdm
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from fintics.ai.util import split_df


class LGBMModel(BaseEstimator, ClassifierMixin):
    """Wrapper around :class:`lightgbm.LGBMClassifier` with trading-specific helpers."""

    def __init__(
        self,
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=1000,
        subsample_for_bin=200000,
        objective: Optional[str] = None,
        class_weight: Optional[str] = None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
    ):
        """Store default LightGBM parameters for later training."""

        self._model = lgb.LGBMClassifier
        self._params = {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample_for_bin': subsample_for_bin,
            'objective': objective,
            'class_weight': class_weight,
            'min_split_gain': min_split_gain,
            'min_child_weight': min_child_weight,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'subsample_freq': subsample_freq,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbose': -1,
        }

    def _get_params(self, trial=None):
        """Return tuned parameters when running with Optuna trial."""

        if trial is not None:
            params = {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            return {**self._params, **params}
        return self._params

    def _metric(self, y_true, y_pred, valid: pd.DataFrame):
        """Custom PL metric used for early stopping."""

        _valid = valid.copy()
        _valid['_y_proba'] = y_pred
        _valid['_y'] = _valid['_y_proba'].STEP(self._threshold)
        pl = (_valid['OpenPL'] * _valid['_y']).sum()
        return 'pl_metric', pl, True

    def fit(
        self,
        train_df: pd.DataFrame,
        cv_splits: int = 1,
        threshold: float = 0.52,
        cv_threshold: float = 0.5,
        cv_verbose: bool = True,
        stopping_rounds: int = 100,
        trial=None,
    ):
        """Fit model using optional cross-validation and custom metrics."""

        self._threshold = threshold
        self._cv_threshold = cv_threshold
        self._cv_splits = cv_splits

        pbar = tqdm(total=self._cv_splits, disable=not cv_verbose)
        if trial is not None:
            stopping_rounds = trial.suggest_int('num_leaves', 10, 1000)
        if self._cv_splits > 1:
            self._models = []
            for _, (train_index, valid_index) in enumerate(
                KFold(n_splits=self._cv_splits, shuffle=True, random_state=42).split(train_df)
            ):

                _train, _valid = train_df.iloc[train_index], train_df.iloc[valid_index]
                _train = _train.loc[_train['Filter']]

                model = self._model(**self._get_params(trial=trial))
                model.fit(
                    _train.X(),
                    _train['y'],
                    eval_set=[(_valid.X(), _valid['y'])],
                    eval_metric=lambda y_true, y_pred: self._metric(y_true, y_pred, _valid),
                    callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
                )
                self._models.append(model)
                pbar.update(1)
        else:
            _train, _valid = split_df(train_df, test_size=0.2, random_state=42)
            _train = _train.loc[_train['Filter']]
            model = self._model(**self._get_params(trial=trial))
            model.fit(
                _train.X(),
                _train['y'],
                eval_set=[(_valid.X(), _valid['y'])],
                eval_metric=lambda y_true, y_pred: self._metric(y_true, y_pred, _valid),
                callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
            )
            self._model = model
            pbar.update(1)
        pbar.close()
        return self

    def predict(self, test_df: pd.DataFrame):
        """Predict labels using trained model(s) and custom thresholds."""

        X = test_df.X()
        test_df['_y'] = 0
        if self._cv_splits > 1:
            for model in self._models:
                test_df['_y_proba'] = model.predict_proba(X)[:, 1]
                test_df['_y'] += test_df['_y_proba'].STEP(self._threshold)
            test_df['_y'] = (test_df['_y'] / len(self._models)).STEP(self._cv_threshold)
        else:
            test_df['_y_proba'] = self._model.predict_proba(X)[:, 1]
            test_df['_y'] = test_df['_y_proba'].STEP(self._threshold)
        return test_df['_y']
