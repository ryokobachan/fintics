"""Data loading and downloading helpers."""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
from tradingview_websocket import TradingViewWebSocket


def get_data_dir() -> Path:
    """
    Get the default data directory for fintics.
    
    Returns ~/.fintics/data/ and creates it if it doesn't exist.
    
    Returns:
        Path: The data directory path
    """
    data_dir = Path.home() / ".fintics" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class Data:
    """
    Data class for loading cached OHLCV data.
    
    Usage:
        # Load from cache (returns pd.DataFrame)
        df = Data('GOOGL')
        df = Data('GOOGL', timeframe='1H')
        
        # Download and cache
        Data.download('GOOGL')
        Data.download(['GOOGL', 'AAPL', 'MSFT'])  # Multiple symbols
        
        # Utilities
        Data.list_cache()
        Data.cache_info('GOOGL')
        Data.clear_cache()
    """

    _cache_dir: Optional[Path] = None

    def __new__(cls, symbol: str, timeframe: str = '1D') -> pd.DataFrame:
        """
        Load data from cache.
        
        Args:
            symbol: Trading symbol (e.g., 'GOOGL', 'AAPL')
            timeframe: Data timeframe (default: '1D')
            
        Returns:
            pd.DataFrame: OHLCV data
            
        Raises:
            FileNotFoundError: If cache doesn't exist. Use Data.download() first.
        """
        timeframe = timeframe.upper()
        cache_path = cls._get_cache_path(symbol, timeframe)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached data for {symbol}/{timeframe}. "
                f"Use Data.download('{symbol}', '{timeframe}') first."
            )
        
        df = pd.read_pickle(cache_path)
        logging.info(f"[{symbol}/{timeframe}] Loaded from cache")
        return df

    @classmethod
    def set_cache_dir(cls, path: Union[str, Path]) -> None:
        """Set a custom cache directory for data storage."""
        cls._cache_dir = Path(path)
        cls._cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Cache directory set to: {cls._cache_dir}")

    @classmethod
    def get_cache_dir(cls) -> Path:
        """Get the current cache directory."""
        if cls._cache_dir is not None:
            return cls._cache_dir
        return get_data_dir()

    @classmethod
    def _get_cache_path(cls, symbol: str, timeframe: str = '1D') -> Path:
        """Get the cache file path for a symbol and timeframe."""
        timeframe = timeframe.upper()
        return cls.get_cache_dir() / f"{symbol}_{timeframe}.pkl"

    @classmethod
    def download(
        cls,
        symbols: Union[str, list[str]],
        timeframe: str = '1D',
        candles: int = 10000,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Download data and save to cache.
        
        Args:
            symbols: Trading symbol or list of symbols
            timeframe: Data timeframe (default: '1D')
            candles: Number of candles to download (default: 10000)
            
        Returns:
            pd.DataFrame if single symbol, dict[symbol, DataFrame] if multiple
            
        Example:
            >>> Data.download('GOOGL')
            >>> Data.download(['GOOGL', 'AAPL', 'MSFT'])
        """
        timeframe = timeframe.upper()
        
        # Normalize to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        results = {}
        single = len(symbols) == 1
        
        pbar = tqdm(symbols, desc="Downloading", disable=single)
        for symbol in pbar:
            if not single:
                pbar.set_postfix_str(symbol)
            try:
                df = cls._export_from_tv(symbol, timeframe, candles)
                cache_path = cls._get_cache_path(symbol, timeframe)
                df.to_pickle(cache_path)
                if single:
                    tqdm.write(f"[{symbol}/{timeframe}] Saved to: {cache_path}")
                results[symbol] = df
            except Exception as e:
                tqdm.write(f"[{symbol}/{timeframe}] Failed: {e}")
                results[symbol] = None
        
        # Return single DataFrame if only one symbol
        if len(results) == 1:
            return list(results.values())[0]
        return results

    @classmethod
    def clear_cache(cls, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol: If provided, clear only this symbol's cache.
            timeframe: If provided with symbol, clear only specific timeframe.
        """
        cache_dir = cls.get_cache_dir()
        
        if symbol is not None:
            if timeframe is not None:
                cache_path = cls._get_cache_path(symbol, timeframe)
                if cache_path.exists():
                    cache_path.unlink()
                    logging.info(f"[{symbol}/{timeframe}] Cache cleared")
            else:
                count = 0
                for pkl_file in cache_dir.glob(f"{symbol}_*.pkl"):
                    pkl_file.unlink()
                    count += 1
                logging.info(f"[{symbol}] Cleared {count} cached files")
        else:
            count = 0
            for pkl_file in cache_dir.glob("*.pkl"):
                pkl_file.unlink()
                count += 1
            logging.info(f"Cleared {count} cached files")

    @classmethod
    def list_cache(cls) -> list[dict]:
        """
        List all cached symbols with their timeframes.
        
        Returns:
            list[dict]: List of {'symbol': ..., 'timeframe': ...}
        """
        cache_dir = cls.get_cache_dir()
        entries = []
        for pkl_file in cache_dir.glob("*.pkl"):
            stem = pkl_file.stem
            if '_' in stem:
                parts = stem.rsplit('_', 1)
                if len(parts) == 2:
                    symbol, timeframe = parts
                    entries.append({'symbol': symbol, 'timeframe': timeframe})
        return sorted(entries, key=lambda x: (x['symbol'], x['timeframe']))

    @classmethod
    def cache_info(cls, symbol: str, timeframe: str = '1D') -> Optional[dict]:
        """
        Get information about a cached symbol.
        
        Returns:
            dict with cache info or None if not cached
        """
        timeframe = timeframe.upper()
        cache_path = cls._get_cache_path(symbol, timeframe)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_pickle(cache_path)
            stat = cache_path.stat()
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'path': str(cache_path),
                'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_at': datetime.fromtimestamp(stat.st_mtime),
                'data_start': df.index[0] if not df.empty else None,
                'data_end': df.index[-1] if not df.empty else None,
                'rows': len(df),
            }
        except Exception as e:
            return {'symbol': symbol, 'timeframe': timeframe, 'error': str(e)}

    @classmethod
    def load(
        cls,
        path: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timeframe: Optional[str] = None,
        spread: float = 0.0,
        remove_no_trade: bool = True,
    ) -> pd.DataFrame:
        """Load OHLCV data from pickle and optionally preprocess it."""
        try:
            df = pd.read_pickle(path)
            df = df.Range(start, end)
            df = df.ohlc(timeframe) if timeframe else df
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['Spread'] = spread
            if remove_no_trade:
                df = df.remove_no_trade()
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {path}: {str(e)}")

    @classmethod
    def export(
        cls,
        symbols: Union[str, list[str]],
        save_path: str,
        timeframe: str = '1D',
        candles: int = 10000,
    ) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Download data from TradingView and save to specified path.
        
        Args:
            symbols: Trading symbol or list of symbols
            save_path: Directory path to save the data
            timeframe: Data timeframe (default: '1D')
            candles: Number of candles to download (default: 10000)
            
        Returns:
            pd.DataFrame if single symbol, dict[symbol, DataFrame] if multiple
        """
        # Normalize to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        os.makedirs(save_path, exist_ok=True)
        results = {}
        single = len(symbols) == 1
        
        pbar = tqdm(symbols, desc="Exporting", disable=single)
        for symbol in pbar:
            if not single:
                pbar.set_postfix_str(symbol)
            try:
                df = cls._export_from_tv(symbol, timeframe, candles)
                path = os.path.join(save_path, f'{symbol}.pkl')
                df.to_pickle(path)
                if single:
                    tqdm.write(f"[{symbol}] Exported to: {path}")
                results[symbol] = df
            except Exception as e:
                tqdm.write(f"[{symbol}] Failed: {e}")
                results[symbol] = None
        
        if len(results) == 1:
            return list(results.values())[0]
        return results

    @classmethod
    def _export_from_tv(cls, symbol: str, timeframe: str = '1H', candles: int = 10000) -> pd.DataFrame:
        """Retrieve raw data via TradingView websocket."""
        try:
            timeframe = timeframe.upper()
            ws = TradingViewWebSocket(symbol, timeframe, candles)
            ws.connect()
            ws.run()
            result_data = ws.result_data

            if not result_data:
                raise ValueError(f"No data received for symbol {symbol}")
            
            _r = len(result_data[0]['v'])
            if _r == 6:
                columns = ['index', 'Open', 'High', 'Low', 'Close', 'Volume']
            elif _r == 5:
                columns = ['index', 'Open', 'High', 'Low', 'Close']
            else:
                raise ValueError(f"Unexpected data format: {_r} columns received for {symbol}")

            df = pd.DataFrame([row['v'] for row in result_data], columns=columns)
            df['index'] = pd.to_datetime(df['index'], unit='s')
            df.set_index('index', inplace=True)
            
            if df.empty or len(df) < 10:
                raise ValueError(f"Insufficient data received for {symbol}")
            
            if timeframe.endswith(('D', 'W', 'M')):
                df.index = df.index.normalize()
                
            return df
        except Exception as e:
            logging.error(f"Failed to export data from TradingView for {symbol}: {str(e)}")
            raise
