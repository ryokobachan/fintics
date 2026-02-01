"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Configure pandas for tests
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Import warnings filter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*tradingview.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*lightgbm.*")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    index = pd.date_range('2023-01-01', periods=100, freq='H')
    return pd.DataFrame({
        'Open': [100 + i * 0.1 for i in range(100)],
        'High': [101 + i * 0.1 for i in range(100)],
        'Low': [99 + i * 0.1 for i in range(100)],
        'Close': [100.5 + i * 0.1 for i in range(100)],
        'Volume': [1000] * 100
    }, index=index)


@pytest.fixture
def sample_strategy():
    """Create a simple strategy for testing."""
    def dummy_strategy(df, **params):
        import pandas as pd
        try:
            from fintics.strategy import Strategy
        except ImportError:
            # Fallback if import fails
            class Strategy(pd.Series):
                def __init__(self, y):
                    super().__init__(y)
                def getY(self, only_buy=True, reverse=False):
                    return self
                def getLastY(self, only_buy=True):
                    return self.iloc[-1] if len(self) > 0 else 0
        
        return Strategy(pd.Series(0, index=df.index))
    
    return dummy_strategy


@pytest.fixture
def sample_ml_data():
    """Create sample ML dataset for testing."""
    return pd.DataFrame({
        'X_feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'X_feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_talib: marks tests that require TA-Lib")
    config.addinivalue_line("markers", "requires_ccxt: marks tests that require CCXT")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add markers based on test names or locations
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "talib" in str(item.function.__name__).lower():
            item.add_marker(pytest.mark.requires_talib)
        if "ccxt" in str(item.function.__name__).lower():
            item.add_marker(pytest.mark.requires_ccxt)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Ensure pandas extensions are loaded if available
    try:
        import fintics.backtest.util  # This registers pandas extensions
    except ImportError:
        pass
    
    yield
    
    # Cleanup after tests
    pass
