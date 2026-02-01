"""Test backtest functionality."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fintics.backtest import Backtest
    from fintics.strategy import Strategy
except ImportError:
    # Fallback to direct imports
    import importlib.util
    
    # Import Backtest
    backtest_spec = importlib.util.spec_from_file_location("backtest", ROOT / "fintics" / "backtest" / "backtest.py")
    backtest_module = importlib.util.module_from_spec(backtest_spec)
    backtest_spec.loader.exec_module(backtest_module)
    Backtest = backtest_module.Backtest
    
    # Import Strategy
    strategy_spec = importlib.util.spec_from_file_location("strategy", ROOT / "fintics" / "strategy" / "strategy.py")
    strategy_module = importlib.util.module_from_spec(strategy_spec)
    strategy_spec.loader.exec_module(strategy_module)
    Strategy = strategy_module.Strategy


def dummy_strategy(df, **params):
    """Simple dummy strategy for testing."""
    df_copy = df.copy()
    df_copy['y'] = 0
    return Strategy(df_copy)


def test_backtest_init_requires_columns():
    """Test that Backtest requires specific columns."""
    df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1]})  # Missing 'Close'
    with pytest.raises(Exception):
        Backtest(df)


def test_backtest_run_generates_performance():
    """Test that backtest run generates performance metrics."""
    index = pd.date_range('2023-01-01', periods=3, freq='h')
    df = pd.DataFrame({
        'Open': [1, 1, 1],
        'High': [1, 1, 1], 
        'Low': [1, 1, 1],
        'Close': [1, 1, 1],
        'Volume': [1, 1, 1]
    }, index=index)
    
    bt = Backtest(df)
    bt.run(strategy=dummy_strategy)
    trades = bt.get_trades()
    perf = bt.get_performance()
    
    assert trades.empty or isinstance(trades, pd.DataFrame)
    assert isinstance(perf, pd.Series)
    assert 'profit' in perf.index


def test_backtest_with_valid_data():
    """Test backtest with valid OHLCV data."""
    index = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Open': [100 + i for i in range(10)],
        'High': [101 + i for i in range(10)],
        'Low': [99 + i for i in range(10)],
        'Close': [100.5 + i for i in range(10)],
        'Volume': [1000] * 10
    }, index=index)
    
    bt = Backtest(df)
    
    # Test that backtest can be instantiated
    assert isinstance(bt, Backtest)
    assert len(bt._df) == 10
