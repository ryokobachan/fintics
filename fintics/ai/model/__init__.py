"""Machine learning model implementations used by fintics."""

try:
    from fintics.ai.model.lgbm import LGBMModel
    from fintics.ai.model.prophet import ProphetModel
except ImportError:
    LGBMModel = None
    ProphetModel = None

__all__ = ['LGBMModel', 'ProphetModel']
