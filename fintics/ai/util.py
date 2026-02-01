"""Utility functions for model evaluation and persistence."""

import dill
from typing import Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def CV(
    df: pd.DataFrame,
    model,
    train_df_filter: Optional[callable] = None,
    n_splits: int = 5,
    skip_trial: Optional[int] = None,
    fit_params: Optional[dict] = None,
    proba_mode: Optional[bool] = None,
    verbose: bool = True,
    adjust_pred_func: Optional[callable] = None,
) -> pd.Series:
    """Perform cross-validation and return concatenated predictions."""

    fit_params = fit_params or {}
    ys = []
    pbar_total = skip_trial if skip_trial is not None and skip_trial < n_splits else n_splits

    with tqdm(total=pbar_total, disable=not verbose) as progress:
        trial = 0
        for i_train, i_test in reversed(list(KFold(n_splits=n_splits).split(df))):
            if skip_trial is not None and trial >= skip_trial:
                break  # stop early if requested

            train_df, test_df = df.iloc[i_train], df.iloc[i_test]
            if train_df_filter is not None:
                train_df = train_df_filter(train_df)  # optional preprocessing of training data

            X_train, X_test, y_train, y_test = train_df.X(), test_df.X(), train_df['y'], test_df['y']
            model.fit(X_train, y_train, **fit_params)
            y_pred = model.predict(X_test, **fit_params) if not proba_mode else model.predict_proba(X_test, **fit_params)[:, 1]

            if adjust_pred_func is not None:
                y_pred = adjust_pred_func(y_pred)  # custom post-processing

            ys = [*y_pred, *ys]
            progress.update(1)
            progress.set_postfix_str(
                f'accuracy: {accuracy_score(y_test, y_pred)}' if not proba_mode else ''
            )
            trial += 1

    ys = [*[0] * (len(df) - len(ys)), *ys]
    return ys


def save_model(model, path: str = 'model.pkl') -> None:
    """Persist the model to ``path`` using dill."""

    with open(path, 'wb') as model_file:
        dill.dump(model, model_file)


def load_model(path: str = 'model.pkl'):
    """Load a model previously saved with :func:`save_model`."""

    with open(path, 'rb') as model_file:
        model = dill.load(model_file)
    return model


def split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: Optional[int] = 42):
    """Split dataframe into train and test parts with optional shuffling."""

    # define random seed
    if random_state is not None:
        np.random.seed(random_state)

    # shuffle the data
    indices = df.index.tolist()
    np.random.shuffle(indices)

    # split the data
    _size = int(len(df) * test_size)
    train_df = df.loc[indices[:-_size]].copy()
    test_df = df.loc[indices[-_size:]].copy()

    return train_df, test_df

