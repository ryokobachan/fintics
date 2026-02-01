from .indicator import Indicator

def _register_pandas_methods():
    """Register indicator methods to pandas Series/DataFrame."""
    import pandas as pd
    for func in Indicator._getSeriesIndicators():
        setattr(pd.Series, func.__name__, func)
    for func in Indicator._getDataframeIndicators():
        setattr(pd.DataFrame, func.__name__, func)

# Auto-register on import (intentional side effect)
_register_pandas_methods()
