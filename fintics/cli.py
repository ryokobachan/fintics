"""Command-line interface for Fintics trading bot."""

# Suppress NumExpr threading messages and other verbose output
import os
import sys
import warnings

os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count() or 8)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 8)
warnings.filterwarnings('ignore')

# Redirect stderr temporarily to suppress numexpr info messages
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    # This will trigger numexpr import if needed
    import pandas as pd
finally:
    # Restore stderr
    sys.stderr = _original_stderr

import argparse
import inspect
import json
import subprocess
from pathlib import Path
from typing import Optional


def get_all_subclasses(cls):
    """Recursively get all subclasses of a class."""
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def get_strategies():
    """Get dictionary of all available strategies with their parameters."""
    # Lazy import
    from fintics.strategy import Strategy
    
    strategies = {}
    for cls in get_all_subclasses(Strategy):
        if cls.__name__ not in ['AverageStrategy', 'CustomStrategy']:
            strategies[cls.__name__] = {
                'class': cls,
                'params': {}
            }
            
            for p in inspect.signature(cls.__init__).parameters:
                if p not in ['self', 'df', 'trial', 'reverse']:
                    param_info = inspect.signature(cls.__init__).parameters[p]
                    default = param_info.default
                    if default is inspect.Parameter.empty:
                        default = None
                    strategies[cls.__name__]['params'][p] = {
                        'default': default
                    }
    
    return strategies


def cmd_help(args):
    """Display help information."""
    help_text = """
Fintics - Trading Bot Framework

USAGE:
    fintics <COMMAND> [OPTIONS]

COMMANDS:
    help, -h                     Show this help message
    strategy list                List all available strategies
    strategy create <name>       Create custom strategy template file
    data list                    List downloaded data
    data download <symbol>       Download data for a symbol
    data delete <symbol>         Delete cached data for a symbol
    run <symbol>                 Run backtest for a symbol
    optimize <symbol>            Optimize strategy for a symbol
    optimize_all <symbol>        Optimize all strategies for a symbol

DATA DOWNLOAD OPTIONS:
    --timeframe <TF>            Timeframe (e.g., 1H, 1D) [default: 1D]

DATA DELETE OPTIONS:
    --timeframe <TF>            Timeframe to delete (e.g., 1H, 1D) [default: all]

BACKTEST/OPTIMIZE OPTIONS:
    --timeframe <TF>            Timeframe (e.g., 1H, 1D) [default: 1D]
    --strategy <NAME>           Strategy name [default: KeepPositionStrategy]
    --strategy-file <PATH>      Path to custom strategy .py file
    --params <JSON>             Strategy parameters as JSON [default: {}]
    --start <DATE>              Start date (e.g., 2022-01-01)
    --end <DATE>                End date (e.g., 2025-12-31)
    --leverage <FLOAT>          Leverage [default: 1.0]
    --spread <FLOAT>            Spread [default: 0.0]
    --feerate <FLOAT>           Fee rate [default: 0.0]
    --onlybuy                   Only allow buy positions [default: True]
    --no-onlybuy                Allow short positions
    --reverse                   Reverse signals [default: False]
    --no-reverse                Don't reverse signals
    --plot                      Show performance plot [default: False]

OPTIMIZE-ONLY OPTIONS:
    --n_trials <INT>            Number of optimization trials [default: 100]
    --target <METRIC>           Optimization target metric [default: profit]
                                (profit, sharperatio, winrate, etc.)

EXAMPLES:
    fintics setup  # Configure shell alias automatically
    fintics strategy create MyStrategy  # Create custom strategy template
    fintics data download GOOGL --timeframe 1H
    fintics data delete GOOGL --timeframe 1H
    fintics run AAPL --strategy RsiStrategy --start 2023-01-01
    fintics run AAPL --strategy-file ./my_strategy.py --plot
    fintics optimize MSFT --n_trials 200 --target sharperatio --plot
    fintics optimize_all BTC --n_trials 50
"""
    print(help_text)


def cmd_strategy_list(args):
    """List all available strategies."""
    strategies = get_strategies()
    
    print("\n" + "=" * 100)
    print(f"{'Strategy Name':<35} {'Default Parameters'}")
    print("-" * 100)
    
    for name, info in sorted(strategies.items()):
        # Format parameters as JSON string (with double quotes)
        if info['params']:
            params_dict = {k: v['default'] for k, v in info['params'].items()}
            params_str = json.dumps(params_dict)
        else:
            params_str = "{}"
        
        print(f"{name:<35} {params_str}")
    
    print("=" * 100)
    print(f"Total: {len(strategies)} strategies\n")


def cmd_data_list(args):
    """List all downloaded data."""
    from fintics.data import Data
    
    cache_list = Data.list_cache()
    
    if not cache_list:
        print("No cached data found.")
        return
    
    print("\nCached Data:")
    print("=" * 80)
    print(f"{'Symbol':<20} {'Timeframe':<15} {'Info'}")
    print("-" * 80)
    
    for entry in cache_list:
        symbol = entry['symbol']
        timeframe = entry['timeframe']
        info = Data.cache_info(symbol, timeframe)
        
        if info and 'rows' in info:
            info_str = f"{info['rows']} rows, {info['file_size_mb']} MB"
        else:
            info_str = "N/A"
        
        print(f"{symbol:<20} {timeframe:<15} {info_str}")
    
    print("=" * 80)
    print(f"Total: {len(cache_list)} datasets")


def load_strategy_from_file(filepath):
    """Load a Strategy class from a Python file.
    
    The filename (without .py) must match the Strategy class name.
    For example: TrendFollowing.py must contain class TrendFollowing(Strategy)
    """
    import importlib.util
    from fintics.strategy import Strategy
    
    filepath = Path(filepath).absolute()
    
    if not filepath.exists():
        print(f"✗ Strategy file not found: {filepath}")
        sys.exit(1)
    
    
    # Get expected strategy name from filename (without .py extension)
    expected_strategy_name = filepath.stem

    # Load the module
    spec = importlib.util.spec_from_file_location("custom_strategy", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find Strategy subclasses in the module
    strategy_classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, Strategy) and 
            obj is not Strategy and
            name != 'Strategy'):
            strategy_classes.append(obj)
    
    if not strategy_classes:
        print(f"✗ No Strategy class found in {filepath}")
        print(f"  Expected a class that inherits from Strategy")
        sys.exit(1)
    
    # Check if the expected strategy name exists
    matching_strategy = None
    for cls in strategy_classes:
        if cls.__name__ == expected_strategy_name:
            matching_strategy = cls
            break
    
    if not matching_strategy:
        print(f"✗ Strategy class name mismatch!")
        print(f"  File name: {filepath.name}")
        print(f"  Expected class: {expected_strategy_name}")
        print(f"  Found classes: {', '.join([c.__name__ for c in strategy_classes])}")
        print(f"\n  Please rename the file to match the class name or vice versa.")
        sys.exit(1)
    
    return matching_strategy


def cmd_strategy_create(args):
    """Create a custom strategy template file."""
    if not args.symbol:  # symbol argument used as strategy name
        print("✗ Strategy name required")
        print("  Usage: fintics strategy create <name>")
        sys.exit(1)
    
    strategy_name = args.symbol
    filename = f"{strategy_name}.py"
    filepath = Path(filename)
    
    if filepath.exists():
        print(f"✗ File already exists: {filename}")
        print(f"  Please choose a different name or delete the existing file.")
        sys.exit(1)
    
    # Get template path
    template_path = Path(__file__).parent / "strategy_template.py"
    
    if not template_path.exists():
        print(f"✗ Template file not found: {template_path}")
        sys.exit(1)
    
    # Read template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders in template
    # STRATEGY_NAME -> user's strategy name (class name)
    # STRATEGY_FILE -> filename without .py
    content = template_content.replace('STRATEGY_NAME', strategy_name)
    content = content.replace('STRATEGY_FILE', filename.replace('.py', ''))
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Created strategy template: {filename}")
    print(f"\nStrategy class: {strategy_name}")
    print(f"\nNext steps:")
    print(f"  1. Edit {filename} to implement your strategy logic")
    print(f"  2. Test with: fintics run SYMBOL --strategy-file {filename}")
    print(f"  3. Optimize with: fintics optimize SYMBOL --strategy-file {filename} --n_trials 100 --plot")


def cmd_data_delete(args):
    """Delete cached data for a symbol."""
    from fintics.data import Data
    
    symbol = args.symbol
    timeframe = args.timeframe
    
    if not timeframe:
        # Delete all timeframes for the symbol
        print(f"Deleting all cached data for {symbol}...")
        
        cache_list = Data.list_cache()
        deleted_count = 0
        
        for entry in cache_list:
            if entry['symbol'] == symbol:
                try:
                    Data.clear_cache(symbol, entry['timeframe'])
                    print(f"  ✓ Deleted: {symbol}/{entry['timeframe']}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {symbol}/{entry['timeframe']}: {e}")
        
        if deleted_count == 0:
            print(f"✗ No cached data found for {symbol}")
        else:
            print(f"\n✓ Deleted {deleted_count} dataset(s) for {symbol}")
    else:
        # Delete specific timeframe
        print(f"Deleting {symbol} ({timeframe})...")
        
        try:
            Data.clear_cache(symbol, timeframe)
            print(f"✓ Successfully deleted {symbol}/{timeframe}")
        except FileNotFoundError:
            print(f"✗ No cached data found for {symbol}/{timeframe}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)


def cmd_data_download(args):
    """Download data for a symbol."""
    from fintics.data import Data
    
    symbol = args.symbol
    timeframe = args.timeframe or '1D'
    
    print(f"Downloading {symbol} ({timeframe})...")
    
    try:
        df = Data.download(symbol, timeframe=timeframe)
        if df is not None:
            print(f"✓ Successfully downloaded {len(df)} rows")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"✗ Failed to download {symbol}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def cmd_run(args):
    """Run backtest for a symbol."""
    from fintics.data import Data
    from fintics.backtest import Backtest
    
    symbol = args.symbol
    timeframe = args.timeframe or '1D'
    strategy_name = args.strategy or 'KeepPositionStrategy'
    
    # Parse params
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON in --params: {e}")
            sys.exit(1)
    
    # Load data
    print(f"Loading {symbol} ({timeframe})...")
    try:
        df = Data(symbol, timeframe=timeframe)
    except FileNotFoundError:
        print(f"✗ No cached data for {symbol}/{timeframe}")
        print(f"  Run: fintics data download {symbol} --timeframe {timeframe}")
        sys.exit(1)
    
    # Filter by date range
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    
    # Get strategy
    if hasattr(args, 'strategy_file') and args.strategy_file:
        # Load custom strategy from file
        strategy_class = load_strategy_from_file(args.strategy_file)
        strategy_name = strategy_class.__name__
        print(f"Loaded custom strategy: {strategy_name}")
    else:
        # Use built-in strategy
        strategies = get_strategies()
        if strategy_name not in strategies:
            print(f"✗ Unknown strategy: {strategy_name}")
            print(f"  Run: fintics strategy list")
            sys.exit(1)
        
        strategy_class = strategies[strategy_name]['class']
    
    # Run backtest
    print(f"Running backtest with {strategy_name}...")
    bt = Backtest(df)
    
    bt.run(
        strategy=strategy_class,
        params=params,
        leverage=args.leverage,
        only_buy=args.onlybuy,
        reverse=args.reverse,
        spread=args.spread,
        fee_rate=args.feerate
    )
    
    # Display results
    perf = bt.get_performance()
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(perf.to_string())
    print("=" * 80)
    
    # Plot if requested
    if hasattr(args, 'plot') and args.plot:
        print("\nGenerating plot...")
        bt.plot()



def cmd_optimize(args):
    """Optimize a strategy for a symbol."""
    from fintics.data import Data
    from fintics.backtest import Backtest
    import pandas as pd

    
    symbol = args.symbol
    timeframe = args.timeframe or '1D'
    strategy_name = args.strategy or 'KeepPositionStrategy'
    n_trials = args.n_trials or 100
    target = args.target or 'profit'
    
    # Load data
    print(f"Loading {symbol} ({timeframe})...")
    try:
        df = Data(symbol, timeframe=timeframe)
    except FileNotFoundError:
        print(f"✗ No cached data for {symbol}/{timeframe}")
        print(f"  Run: fintics data download {symbol} --timeframe {timeframe}")
        sys.exit(1)
    
    # Filter by date range
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    
    # Get strategy
    if hasattr(args, 'strategy_file') and args.strategy_file:
        # Load custom strategy from file
        strategy_class = load_strategy_from_file(args.strategy_file)
        strategy_name = strategy_class.__name__
        print(f"Loaded custom strategy: {strategy_name}")
    else:
        # Use built-in strategy
        strategies = get_strategies()
        if strategy_name not in strategies:
            print(f"✗ Unknown strategy: {strategy_name}")
            print(f"  Run: fintics strategy list")
            sys.exit(1)
        
        strategy_class = strategies[strategy_name]['class']
    
    # Run optimization
    print(f"Optimizing {strategy_name} for {n_trials} trials...")
    print(f"Target metric: {target}")
    
    bt = Backtest(df)
    result = bt.optimize(
        strategy=strategy_class,
        n_trials=n_trials,
        target=target,
        leverage=args.leverage,
        only_buy=args.onlybuy,
        reverse=args.reverse,
        spread=args.spread,
        fee_rate=args.feerate
    )
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Results is a DataFrame with optimization history
    if isinstance(result, pd.DataFrame) and not result.empty:
        best_result = result.iloc[0]  # First row is the best result
        print(f"\nBest {target}: {best_result.get(target, 'N/A')}")
        print(f"Best params: {best_result.get('params', {})}")
        
        print("\n" + "-" * 80)
        print("Best Result Summary:")
        print("-" * 80)
        
        # Display all available columns for the best result
        summary_cols = [
            'strategy', 'params',
            'profit', 'growth', 'avg_profit', 'win_rate', 'n_trades', 'efficiency',
            'PD', 'PDE', 'PD_vs_Growth', 'PDE_vs_Growth',
            'max_dd', 'max_dd_rate', 'max_dd_duration', 'dd_avg',
            'sqn', 'profitfactor', 'sharperatio', 'sortinoratio',
            'profit/best_profit'
        ]
        
        # Only include columns that actually exist in the result
        available_cols = [col for col in summary_cols if col in result.columns]
        
        # Display best result in detail
        for col in available_cols:
            if col not in ['strategy', 'params']:
                value = best_result.get(col, 'N/A')
                if isinstance(value, (int, float)):
                    print(f"  {col:<20}: {value:>15.4f}")
                else:
                    print(f"  {col:<20}: {value}")
        
        # Show top 10 results in table format with ALL columns
        print("\n" + "-" * 80)
        print("Top 10 Results (All Metrics):")
        print("-" * 80)
        
        # Display all available columns
        display_df = result.head(10)
        print(display_df.to_string(index=False))

    else:
        print("No optimization results available")
    
    print("=" * 80)
    
    # Plot if requested
    if hasattr(args, 'plot') and args.plot:
        print("\nGenerating plot...")
        bt.plot()



def cmd_optimize_all(args):
    """Optimize all strategies for a symbol."""
    import pandas as pd
    from fintics.data import Data
    from fintics.backtest import Backtest
    
    symbol = args.symbol
    timeframe = args.timeframe or '1D'
    n_trials = args.n_trials or 100
    target = args.target or 'profit'
    
    # Load data
    print(f"Loading {symbol} ({timeframe})...")
    try:
        df = Data(symbol, timeframe=timeframe)
    except FileNotFoundError:
        print(f"✗ No cached data for {symbol}/{timeframe}")
        print(f"  Run: fintics data download {symbol} --timeframe {timeframe}")
        sys.exit(1)
    
    # Filter by date range
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]
    
    # Get all strategies
    strategies = get_strategies()
    
    print(f"Optimizing {len(strategies)} strategies for {n_trials} trials each...")
    print(f"Target metric: {target}")
    
    bt = Backtest(df)
    results = bt.optimize_all(
        n_trials=n_trials,
        target=target,
        leverage=args.leverage,
        only_buy=args.onlybuy,
        reverse=args.reverse,
        spread=args.spread,
        fee_rate=args.feerate
    )
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS (ALL STRATEGIES)")
    print("=" * 80)
    
    if isinstance(results, pd.DataFrame) and not results.empty:
        # Show summary of all strategies
        print(f"\nOptimized {len(results)} strategies\n")
        print(results.to_string())
    elif isinstance(results, dict):
        # Handle dict format (strategy_name: performance)
        for strategy_name, perf in results.items():
            print(f"\n{strategy_name}:")
            if isinstance(perf, pd.Series):
                print(f"  Best {target}: {perf.get(target, 'N/A')}")
                print(f"  Profit: {perf.get('profit', 'N/A')}")
                print(f"  Params: {perf.get('params', {})}")
            else:
                print(f"  {perf}")
    else:
        print("No optimization results available")
    
    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fintics - Trading Bot Framework',
        add_help=False
    )
    
    parser.add_argument('command', nargs='?', help='Command to run')
    parser.add_argument('subcommand', nargs='?', help='Subcommand')
    parser.add_argument('symbol', nargs='?', help='Trading symbol')
    
    # Data download options
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1H, 1D)')
    
    # Backtest/optimize options
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--strategy-file', type=str, help='Path to custom strategy .py file')
    parser.add_argument('--params', type=str, help='Strategy parameters as JSON')
    parser.add_argument('--start', type=str, help='Start date')
    parser.add_argument('--end', type=str, help='End date')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage')
    parser.add_argument('--spread', type=float, default=0.0, help='Spread')
    parser.add_argument('--feerate', type=float, default=0.0, help='Fee rate')
    parser.add_argument('--onlybuy', action='store_true', default=True, help='Only buy')
    parser.add_argument('--no-onlybuy', dest='onlybuy', action='store_false', help='Allow short')
    parser.add_argument('--reverse', action='store_true', default=False, help='Reverse signals')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false', help='No reverse')
    parser.add_argument('--plot', action='store_true', default=False, help='Show performance plot')
    
    # Optimize options
    parser.add_argument('--n_trials', type=int, help='Number of optimization trials')
    parser.add_argument('--target', type=str, help='Optimization target metric')
    
    # Help flag
    parser.add_argument('-h', '--help', action='store_true', help='Show help')
    
    args = parser.parse_args()
    
    # Handle help
    if args.help or args.command in ['help', None]:
        cmd_help(args)
        return
    
    # Route commands
    try:
        
        if args.command == 'strategy':
            if args.subcommand == 'list':
                cmd_strategy_list(args)
            elif args.subcommand == 'create':
                cmd_strategy_create(args)
            else:
                print(f"✗ Unknown subcommand: strategy {args.subcommand}")
                print("  Try: fintics strategy list  OR  fintics strategy create <name>")
                sys.exit(1)
        
        elif args.command == 'data':
            if args.subcommand == 'list':
                cmd_data_list(args)
            elif args.subcommand == 'download':
                if not args.symbol:
                    print("✗ Symbol required for data download")
                    print("  Usage: fintics data download <symbol>")
                    sys.exit(1)
                cmd_data_download(args)
            elif args.subcommand == 'delete':
                if not args.symbol:
                    print("✗ Symbol required for data delete")
                    print("  Usage: fintics data delete <symbol>")
                    sys.exit(1)
                cmd_data_delete(args)
            else:
                print(f"✗ Unknown subcommand: data {args.subcommand}")
                print("  Try: fintics data list  OR  fintics data download <symbol>  OR  fintics data delete <symbol>")
                sys.exit(1)
        
        elif args.command == 'run':
            if not args.subcommand:
                print("✗ Symbol required for run")
                print("  Usage: fintics run <symbol>")
                sys.exit(1)
            args.symbol = args.subcommand
            cmd_run(args)
        
        elif args.command == 'optimize':
            if not args.subcommand:
                print("✗ Symbol required for optimize")
                print("  Usage: fintics optimize <symbol>")
                sys.exit(1)
            args.symbol = args.subcommand
            cmd_optimize(args)
        
        elif args.command == 'optimize_all':
            if not args.subcommand:
                print("✗ Symbol required for optimize_all")
                print("  Usage: fintics optimize_all <symbol>")
                sys.exit(1)
            args.symbol = args.subcommand
            cmd_optimize_all(args)
        
        else:
            print(f"✗ Unknown command: {args.command}")
            print("  Run: fintics help")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
