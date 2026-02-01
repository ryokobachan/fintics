"""Feature selection utilities for machine learning datasets."""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """Provide simple feature selection routines."""

    @classmethod
    def CORR(cls, df: pd.DataFrame, min_corr: float = 0.100) -> pd.DataFrame:
        """Drop X_* columns that have absolute correlation below ``min_corr`` with target ``y``."""

        # Get X_ columns and add y column for correlation calculation
        x_columns = df.X()
        corr_df = x_columns.assign(y=df['y'])
        
        # Compute correlations with target y
        corr = corr_df.corr()['y']
        
        # Remove the correlation of y with itself (which is always 1.0)
        corr = corr.drop('y', errors='ignore')
        
        # Handle NaN correlations (e.g., constant features) by treating them as 0 correlation
        corr = corr.fillna(0)
        
        # Identify features with absolute correlation below threshold
        weak_features = corr[corr.abs() < min_corr].index
        
        # Drop weakly correlated features from original dataframe
        return df.drop(weak_features, axis=1)

    @classmethod
    def VarianceThreshold(cls, df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """Remove X_* columns whose variance is below ``threshold``."""

        _df = df.X()  # only use feature columns
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(_df)
        # get_support() returns True for features to KEEP (high variance)
        # We want to drop features with low variance, so we use ~get_support()
        low_variance_mask = ~sel.get_support(indices=False)
        col_to_drop = _df.columns[low_variance_mask]  # low-variance columns to drop
        return df.drop(col_to_drop, axis=1)

