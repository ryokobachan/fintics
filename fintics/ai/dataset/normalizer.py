"""Utility helpers for scaling feature columns in a dataframe."""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import pandas as pd


class Normalizer:
    """Collection of methods to apply different scaling strategies."""

    @classmethod
    def _normalize(cls, scaler, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the given scaler and return a scaled dataframe."""

        scaled_data = scaler.fit_transform(df)  # scale the features
        # keep original index and columns when converting back to DataFrame
        scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        return scaled_df

    @classmethod
    def _combine(cls, df: pd.DataFrame, X_df: pd.DataFrame) -> pd.DataFrame:
        """Combine scaled feature columns back into the original dataframe."""

        return X_df.combine_first(df)

    @classmethod
    def MinMaxScaler(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Scale X_* columns to [0, 1] range."""

        scaler = MinMaxScaler()
        X_df = cls._normalize(scaler, df.filter(like='X'))
        return cls._combine(df, X_df)

    @classmethod
    def StandardScaler(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize X_* columns to zero mean and unit variance."""

        scaler = StandardScaler()
        X_df = cls._normalize(scaler, df.filter(like='X'))
        return cls._combine(df, X_df)

    @classmethod
    def L1(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize X_* columns using L1 norm."""

        scaler = Normalizer(norm='l1')
        X_df = cls._normalize(scaler, df.filter(like='X'))
        return cls._combine(df, X_df)

    @classmethod
    def L2(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize X_* columns using L2 norm."""

        scaler = Normalizer(norm='l2')
        X_df = cls._normalize(scaler, df.filter(like='X'))
        return cls._combine(df, X_df)

    @classmethod
    def RobustScaler(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Scale X_* columns using the robust scaler (median/IQR)."""

        scaler = RobustScaler()
        X_df = cls._normalize(scaler, df.filter(like='X'))
        return cls._combine(df, X_df)

