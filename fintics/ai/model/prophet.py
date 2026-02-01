"""Prophet based regresser."""

from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin


class ProphetModel(BaseEstimator, RegressorMixin):
    """
    A wrapper class for the Prophet model to make it compatible with the scikit-learn interface.
    
    This class automatically handles:
    - Conversion of a DatetimeIndex to the 'ds' column required by Prophet.
    - Identification and addition of extra regressors from columns containing 'X' in their name.
    """
    def __init__(self, growth='linear', changepoints=None, n_changepoints=25,
                 changepoint_range=0.8, yearly_seasonality='auto',
                 weekly_seasonality='auto', daily_seasonality='auto',

                 holidays=None, seasonality_mode='additive',
                 seasonality_prior_scale=10.0, holidays_prior_scale=10.0,
                 changepoint_prior_scale=0.05, mcmc_samples=0, interval_width=0.80,
                 uncertainty_samples=1000, stan_backend=None):
        """
        Initialize the model with Prophet's hyperparameters.
        """
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        
        self.model = None
        self.regressors_ = [] # To store the names of the extra regressors

    def fit(self, X, y):
        """
        Fit the Prophet model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with a DatetimeIndex and feature columns (extra regressors).
            Feature columns should contain 'X' in their name (e.g., 'X_feature1').
        y : pd.Series or np.array
            Target values.
        """
        # --- Data Preparation ---
        # Work on a copy to avoid modifying the original dataframe
        train_df = X.copy()
        
        # Create the 'ds' column from the dataframe's index
        train_df['ds'] = train_df.index
        train_df['y'] = y

        # --- Model Initialization and Training ---
        self.model = Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            mcmc_samples=self.mcmc_samples,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend
        )
        
        # Add the identified regressors to the model
        for regressor in self.regressors_:
            self.model.add_regressor(regressor)
        
        self.model.fit(train_df)
        
        return self

    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with a DatetimeIndex and future values for the regressors.
            
        Returns
        -------
        np.array
            Predicted values ('yhat').
        """
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call 'fit' first.")
        
        # --- Data Preparation for Prediction ---
        future_df = X.copy()
        future_df['ds'] = future_df.index
        
        # --- Prediction ---
        forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values
    