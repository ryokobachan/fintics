# Fintics - Trading Bot Framework

Advanced trading bot framework with backtesting and strategy optimization capabilities

## Table of Contents
- [Installation](#installation)
- [Quick Setup](#-quick-setup-recommended)
- [Command List](#command-list)
- [Custom Strategy Creation](#custom-strategy-creation-guide)
- [Available Strategies](#available-strategies)
- [Practical Examples](#practical-examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Features](#-features)
- [Development](#-development)

## Installation

### From PyPI (Recommended)

```bash
pip install fintics
```

### From Source (For Development)

```bash
# Navigate to project directory
cd /path/to/Fintics

# Install (development mode)
pip install -e .

# Or standard installation
pip install .
```

After installation, the `fintics` command is available globally.

```bash
fintics help
fintics data list
fintics run AAPL --strategy RsiStrategy
```

## Usage

**Standard Usage**:
```bash
fintics <COMMAND> [OPTIONS]
```

**Alternative (Run as Python module)**:
```bash
python -m fintics.cli <COMMAND> [OPTIONS]
```

## Command List

### Help

```bash
fintics help
fintics -h
```

### Strategy Management

**List Available Strategies:**
```bash
fintics strategy list
```

**Create Custom Strategy Template:**
```bash
fintics strategy create MyStrategy

# Creates MyStrategy.py
# - Optimization-ready (trial argument support)
# - Sample RSI strategy implementation
# - Comprehensive documentation included
```

### Data Management

**List Downloaded Data:**
```bash
fintics data list
```

**Download Data:**
```bash
fintics data download <symbol> [--timeframe <TF>]

# Examples:
fintics data download GOOGL
fintics data download AAPL --timeframe 1H
```

**Delete Data:**
```bash
# Delete specific timeframe
fintics data delete <symbol> --timeframe <TF>

# Delete all timeframes
fintics data delete <symbol>

# Examples:
fintics data delete GOOGL --timeframe 1H
fintics data delete AAPL  # Delete all timeframes
```

### Run Backtest

**Basic Execution:**
```bash
fintics run <symbol> [OPTIONS]

# Examples:
fintics run AAPL
fintics run MSFT --strategy RsiStrategy --start 2023-01-01
fintics run GOOGL --strategy BollingerBandsStrategy --params '{"t": 20, "std": 2.5}'
```

**Using Custom Strategy File:**
```bash
fintics run <symbol> --strategy-file <PATH> [OPTIONS]

# Examples:
fintics run AAPL --strategy-file ./MyStrategy.py
fintics run MSFT --strategy-file ./custom_strategies/advanced.py --plot
```

**Display Results Plot:**
```bash
fintics run AAPL --strategy RsiStrategy --plot
```

**Options:**
- `--timeframe <TF>` - Timeframe (e.g., 1H, 1D) [default: 1D]
- `--strategy <NAME>` - Built-in strategy name [default: KeepPositionStrategy]
- `--strategy-file <PATH>` - Path to custom strategy .py file
- `--params <JSON>` - Strategy parameters (JSON format) [default: {}]
- `--start <DATE>` - Start date (e.g., 2022-01-01)
- `--end <DATE>` - End date (e.g., 2025-12-31)
- `--leverage <FLOAT>` - Leverage [default: 1.0]
- `--spread <FLOAT>` - Spread [default: 0.0]
- `--feerate <FLOAT>` - Fee rate [default: 0.0]
- `--onlybuy` / `--no-onlybuy` - Long positions only [default: True]
- `--reverse` / `--no-reverse` - Reverse signals [default: False]
- `--plot` - Display performance chart [default: False]

### Optimization

**Optimize Single Strategy:**
```bash
fintics optimize <symbol> [OPTIONS]

# Examples:
fintics optimize AAPL --strategy RsiStrategy
fintics optimize MSFT --strategy MacdStrategy --n_trials 200 --target sharperatio
```

**Optimize Custom Strategy File:**
```bash
fintics optimize <symbol> --strategy-file <PATH> [OPTIONS]

# Examples:
fintics optimize AAPL --strategy-file ./MyStrategy.py --n_trials 100
fintics optimize GOOGL --strategy-file ./advanced.py --n_trials 200 --plot
```

**Optimize All Strategies:**
```bash
fintics optimize_all <symbol> [OPTIONS]

# Examples:
fintics optimize_all BTCUSD --n_trials 50
fintics optimize_all GOOGL --n_trials 100 --target profit

# Note: --plot option is not available for optimize_all
```

**Optimization Options:**
- `--n_trials <INT>` - Number of optimization trials [default: 100]
- `--target <METRIC>` - Target optimization metric [default: profit]
  - Available: `profit`, `sharperatio`, `winrate`, etc.
- `--plot` - Plot optimization results (not available for optimize_all)

## Custom Strategy Creation Guide

### 1. Generate Template File

```bash
fintics strategy create MyStrategy
```

Generated `MyStrategy.py`:

```python
import pandas as pd
from fintics.strategy import Strategy
from fintics.indicator import Indicator

class MyStrategy(Strategy):
    def __init__(self, df: pd.DataFrame, rsi_period: int = 14, ma_short: int = 10, 
                 ma_long: int = 30, trial=None, reverse: bool = False):
        # Parameter suggestions for optimization
        if trial:
            rsi_period = trial.suggest_int('rsi_period', 5, 30)
            ma_short = trial.suggest_int('ma_short', 5, 20)
            ma_long = trial.suggest_int('ma_long', 20, 50)
        
        # Calculate indicators
        rsi = Indicator.RSI(df['Close'], timeperiod=rsi_period)
        ma_short_values = Indicator.SMA(df['Close'], timeperiod=ma_short)
        ma_long_values = Indicator.SMA(df['Close'], timeperiod=ma_long)
        
        # Generate signals
        df['y'] = 0
        
        # Buy signal: MA crossover AND RSI < 70
        buy_condition = (
            (ma_short_values > ma_long_values) & 
            (ma_short_values.shift(1) <= ma_long_values.shift(1)) & 
            (rsi < 70)
        )
        df.loc[buy_condition, 'y'] = 1
        
        # Sell signal: MA crossover OR RSI > 70
        sell_condition = (
            (ma_short_values < ma_long_values) & 
            (ma_short_values.shift(1) >= ma_long_values.shift(1)) | 
            (rsi > 70)
        )
        df.loc[sell_condition, 'y'] = -1
        
        super().__init__(df)
```

### 2. Test Strategy

```bash
fintics run AAPL --strategy-file MyStrategy.py
```

### 3. Optimize Parameters

```bash
fintics optimize AAPL --strategy-file MyStrategy.py --n_trials 200 --plot
```

### 4. Run with Optimal Parameters

```bash
fintics run AAPL --strategy-file MyStrategy.py --params '{"rsi_period": 25, "ma_short": 15, "ma_long": 45}' --plot
```

## Available Strategies

List all strategies:
```bash
fintics strategy list
```

49 built-in strategies available, including:
- `RsiStrategy` - RSI-based strategy
- `MacdStrategy` - MACD strategy
- `BollingerBandsStrategy` - Bollinger Bands strategy
- `StochasticStrategy` - Stochastic oscillator strategy
- `IchimokuCloudStrategy` - Ichimoku Cloud strategy
- And more...

Default parameters for each strategy are displayed in JSON format:

```
====================================================================================================
Strategy Name                       Default Parameters
----------------------------------------------------------------------------------------------------
RsiStrategy                        {"t": 14}
MacdStrategy                       {"fast": 12, "slow": 26, "signal": 9}
BollingerBandsStrategy             {"t": 4, "std": 2, "ma_type": 2}
ParabolicSarStrategy              {"acceleration": 0.02, "maximum": 0.2}
====================================================================================================
Total: 49 strategies
```

## Practical Examples

### Example 1: Download Data and Run Backtest

```bash
# Download data
fintics data download AAPL

# Backtest with RSI strategy
fintics run AAPL --strategy RsiStrategy --plot

# Specify date range
fintics run AAPL --strategy RsiStrategy --start 2023-01-01 --end 2024-12-31 --plot
```

### Example 2: Create and Optimize Custom Strategy

```bash
# Create strategy template
fintics strategy create TrendFollowing

# Edit and implement logic
# nano TrendFollowing.py

# Test run
fintics run MSFT --strategy-file TrendFollowing.py

# Optimize parameters
fintics optimize MSFT --strategy-file TrendFollowing.py --n_trials 300 --target sharperatio --plot

# Run with optimal parameters
fintics run MSFT --strategy-file TrendFollowing.py --params '{"ma_short": 15, "ma_long": 45}' --plot
```

### Example 3: Multi-Timeframe Analysis

```bash
# Download hourly data
fintics data download BTCUSD --timeframe 1H

# Backtest on 1H timeframe
fintics run BTCUSD --timeframe 1H --strategy MacdStrategy --plot

# Compare with daily timeframe
fintics run BTCUSD --timeframe 1D --strategy MacdStrategy --plot
```

### Example 4: Data Management

```bash
# Check downloaded data
fintics data list

# Delete unnecessary data
fintics data delete GOOGL --timeframe 1H

# Delete all data for a symbol
fintics data delete OLD_SYMBOL
```

## Troubleshooting

### Q: "No cached data" error occurs

```bash
# Download the data first
fintics data download <SYMBOL> --timeframe 1D
```

### Q: Custom Strategy not found

- Verify file path is correct
- Ensure Strategy class is defined
- Check `from fintics.strategy import Strategy` is imported
- **Important:** File name must match class name (e.g., `MyStrategy.py` must contain `class MyStrategy`)

### Q: Optimization is slow

- Reduce `--n_trials` (start with 50-100)
- Limit data range with `--start`/`--end`

### Q: setup command cannot detect shell

Supported shells: zsh, bash, fish

Manually add alias:
```bash
# Add to ~/.zshrc or ~/.bashrc
alias fintics='python -m fintics.cli'
```

## Advanced Usage

### Leverage and Spread Consideration

```bash
fintics run BTCUSD --strategy RsiStrategy --leverage 2.0 --spread 0.001 --feerate 0.0005
```

### Allow Short Positions

```bash
fintics run AAPL --strategy RsiStrategy --no-onlybuy
```

### Reverse Signals

```bash
fintics run MSFT --strategy RsiStrategy --reverse
```

### Batch Optimize All Strategies

```bash
fintics optimize_all GOOGL --n_trials 100 --target sharperatio
```

### Parameter Specification

You can copy and paste JSON format displayed in strategy list:

```bash
# Check strategy list
fintics strategy list | grep "Parabolic"
# Output: ParabolicSarStrategy {"acceleration": 0.02, "maximum": 0.2}

# Copy and use directly
fintics run AAPL --strategy ParabolicSarStrategy --params '{"acceleration": 0.02, "maximum": 0.2}'
```

## 🎉 Features

- ✅ **49 Built-in Trading Strategies** - Ready-to-use professional strategies
- ✅ **Powerful Backtest Engine** - Comprehensive backtesting with real-world constraints
- ✅ **Optuna-based Optimization** - Advanced parameter optimization
- ✅ **TradingView Data Integration** - Download data via TradingView
- ✅ **Custom Strategy Support** - Create and optimize your own strategies
- ✅ **Comprehensive Performance Metrics** - 20+ metrics including Sharpe ratio, profit factor, drawdown, etc.
- ✅ **Command-line Interface** - Easy-to-use CLI with intuitive commands
- ✅ **Plot Visualization** - Visual performance analysis with charts
- ✅ **Multi-timeframe Support** - Backtest on various timeframes (1H, 1D, etc.)
- ✅ **Data Management** - Download, list, and delete cached data

### New Features

#### Custom Strategy Creation
```bash
fintics strategy create MyStrategy
```
- Generates optimization-ready template
- File name must match class name
- Includes sample implementation with multiple indicators

#### Enhanced Optimization Results
- Display all 20+ BacktestResult metrics
- Top 10 results comparison table
- Detailed best result summary

Example output:
```
================================================================================
OPTIMIZATION RESULTS
================================================================================

Best profit: 106106.7226
Best params: {"t": 102}

--------------------------------------------------------------------------------
Best Result Summary:
--------------------------------------------------------------------------------
  profit              :     106106.7226
  growth              :     158769.6156
  avg_profit          :        614.2976
  win_rate            :          0.2617
  n_trades            :            149
  efficiency          :          0.6776
  PD                  :     343721.1616
  PDE                 :     247719.0732
  max_dd_rate         :         -0.3087
  sharperatio         :          4.5104
  profitfactor        :          5.1147
  ...and more metrics
================================================================================
```

#### Flexible Data Management
```bash
# Delete specific timeframe
fintics data delete GOOGL --timeframe 1H

# Delete all timeframes for a symbol
fintics data delete GOOGL
```

#### Performance Visualization
```bash
# Add --plot to see charts
fintics run AAPL --strategy RsiStrategy --plot
fintics optimize MSFT --strategy MacdStrategy --n_trials 100 --plot
```

## Tips

- **Use Plots**: Use `--plot` option to visualize results
- **Start Small**: Begin optimization with `--n_trials 50`
- **Limit Date Range**: Use `--start`/`--end` to limit testing period
- **Custom Strategies**: Improve based on existing built-in strategies
- **Data Cleanup**: Regularly delete old data to save space

## Summary

Fintics CLI provides:

✅ **49 Built-in Strategies** - High-quality strategies ready to use  
✅ **Custom Strategy Creation** - Easy implementation from templates  
✅ **Automatic Optimization** - Advanced parameter optimization with Optuna  
✅ **Visualization** - Intuitive understanding with plot features  
✅ **Flexible Data Management** - Download, delete, multi-timeframe support  
✅ **User-friendly CLI** - Intuitive command structure

## 🔧 Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black fintics/
isort fintics/
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📞 Support

For help:
```bash
fintics help
```

For issues and feature requests, please visit the project repository.
