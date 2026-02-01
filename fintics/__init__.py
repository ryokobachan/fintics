"""Fintics - Financial backtesting and strategy development library."""

__version__ = "3.0.0"

# === Setup ===
import os
import sys
import warnings
from logging import basicConfig, INFO
import io

# Suppress NumExpr threading messages
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count() or 8)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 8)

# Temporarily redirect stdout/stderr to suppress numexpr info
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

sys.dont_write_bytecode = True
warnings.simplefilter('ignore')
basicConfig(level=INFO, format='%(message)s')

# === Pandas settings ===
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# === Core ===
from fintics.data import Data
from fintics.indicator import Indicator
from fintics.backtest import Backtest, RealtimeBacktest, optimize
from fintics.strategy import Strategy

# Restore stdout/stderr after imports
sys.stdout = _original_stdout
sys.stderr = _original_stderr

# === AI (optional) ===
try:
    from fintics.ai import Dataset, BasicDataset, LGBMModel, ProphetModel, CV, save_model, load_model
except ImportError:
    pass
