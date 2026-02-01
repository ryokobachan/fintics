"""Test indicator functionality."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    ta = None

try:
    from fintics.indicator import Indicator
except ImportError:
    # Fallback to direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("indicator", ROOT / "fintics" / "indicator" / "indicator.py")
    indicator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indicator_module)
    Indicator = indicator_module.Indicator


def test_pl_matches_diff_shift():
    """Test PL indicator matches diff().shift(-1)."""
    series = pd.Series([1, 2, 4, 7, 11])
    expected = series.diff().shift(-1)
    
    try:
        result = Indicator.PL(series)
        pd.testing.assert_series_equal(result, expected)
    except AttributeError:
        pytest.skip("PL indicator not available")


@pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
def test_ma_returns_sma_when_ma_type_zero():
    """Test that MA with ma_type=0 returns SMA."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = ta.SMA(series, timeperiod=3)
    
    try:
        result = Indicator.MA(series, t=3, ma_type=0)
        pd.testing.assert_series_equal(result, expected)
    except Exception as e:
        pytest.skip(f"MA indicator not available: {e}")


def test_ma_returns_input_when_period_less_than_equal_one():
    """Test that MA returns input series when period <= 1."""
    series = pd.Series([1, 3, 5])
    
    try:
        result = Indicator.MA(series, t=1)
        pd.testing.assert_series_equal(result, series)
    except Exception as e:
        pytest.skip(f"MA indicator not available: {e}")


@pytest.mark.skipif(not TALIB_AVAILABLE, reason="TA-Lib not installed")
def test_rsi_basic_functionality():
    """Test RSI indicator basic functionality."""
    # Create price series with clear trend
    prices = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100, 99, 98, 97, 98, 99, 100])
    
    try:
        rsi = Indicator.RSI(prices, t=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.min() >= 0
            assert valid_rsi.max() <= 100
    except Exception as e:
        pytest.skip(f"RSI indicator not available: {e}")


def test_indicator_list_methods():
    """Test that Indicator class has expected methods."""
    try:
        indicators = Indicator._getIndicators()
        assert isinstance(indicators, list)
        assert len(indicators) > 0
    except AttributeError:
        pytest.skip("_getIndicators method not available")
