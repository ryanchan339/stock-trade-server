"""Download and cache daily OHLCV history from Yahoo Finance."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _cache_path(cache_dir: str | Path, ticker: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker}.pkl"


def download_history(
    ticker: str,
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data_cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame for one ticker, using a local pickle cache.

    Columns: open, high, low, close, volume. Index: tz-naive DatetimeIndex.
    """
    cache = _cache_path(cache_dir, ticker)
    if cache.exists() and not refresh:
        df = pd.read_pickle(cache)
        # Top up with any newer rows since the cache was written.
        last = df.index.max()
        fresh_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp.today().normalize() > last:
            new = _yf_download(ticker, fresh_start, end)
            if not new.empty:
                df = pd.concat([df, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_pickle(cache)
        return df

    df = _yf_download(ticker, start, end)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.to_pickle(cache)
    return df


def _yf_download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,      # split/dividend adjusted prices
        progress=False,
        threads=False,
    )
    if raw.empty:
        return raw
    # yfinance returns a column MultiIndex when given a single ticker in newer
    # versions; flatten it to plain lower-case OHLCV.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw = raw[["open", "high", "low", "close", "volume"]]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return raw.dropna()


def download_universe(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data_cache",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download history for every ticker; skip any that fail."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = download_history(t, start, end, cache_dir, refresh)
            log.info("Loaded %s (%d rows)", t, len(out[t]))
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't kill the run
            log.warning("Failed to load %s: %s", t, e)
    return out
