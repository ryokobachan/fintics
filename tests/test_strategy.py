"""Test strategy functionality."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fintics.strategy import Strategy
except ImportError:
    # Fallback to direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("strategy", ROOT / "fintics" / "strategy" / "strategy.py")
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    Strategy = strategy_module.Strategy


def test_strategy_getY_and_getLastY():
    """Test Strategy getY and getLastY methods."""
    df = pd.DataFrame({'y': [1, -1, 0]}, index=[0, 1, 2])
    s = Strategy(df)
    assert s.getY().tolist() == [1, 0, 0]
    assert s.getY(only_buy=False).tolist() == [1, -1, 0]
    assert s.getLastY() == 0


def test_strategy_reverse_flag():
    """Test Strategy reverse functionality."""
    df = pd.DataFrame({'y': [1, -1, 0]}, index=[0, 1, 2])
    s = Strategy(df)
    y_normal = s.getY(only_buy=False, reverse=False)
    y_reversed = s.getY(only_buy=False, reverse=True)
    
    assert y_reversed.tolist() == [-1, 1, 0]  # Should be reversed


def test_strategy_inheritance():
    """Test that Strategy can be properly subclassed."""
    class TestStrategy(Strategy):
        def __init__(self, values):
            df = pd.DataFrame({'y': values})
            super().__init__(df)
    
    test_strat = TestStrategy([0.5, -0.5, 0])
    assert isinstance(test_strat, Strategy)
    assert test_strat.getLastY() == 0
