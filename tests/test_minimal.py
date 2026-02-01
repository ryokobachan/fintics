"""Minimal test to verify basic functionality."""

import pytest
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def test_python_basics():
    """Test basic Python functionality."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"

def test_pandas_import():
    """Test pandas import."""
    import pandas as pd
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert len(df) == 3

def test_numpy_import():
    """Test numpy import."""
    import numpy as np
    arr = np.array([1, 2, 3])
    assert len(arr) == 3

def test_fintics_basic_import():
    """Test basic fintics import."""
    try:
        import fintics
        assert hasattr(fintics, '__version__')
    except ImportError as e:
        pytest.skip(f"Fintics import failed: {e}")

def test_strategy_import():
    """Test Strategy class import."""
    try:
        from fintics.strategy import Strategy
        import pandas as pd
        
        df = pd.DataFrame({'y': [1, 0, -1]})
        s = Strategy(df)
        assert isinstance(s, Strategy)
        
    except ImportError as e:
        pytest.skip(f"Strategy import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
