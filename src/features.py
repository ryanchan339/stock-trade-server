"""Feature engineering: turn OHLCV history into model inputs and labels.

Every feature is computed from data available *at or before* the row's date, so
there is no lookahead. The label, by contrast, peeks into the future and is only
used for training/evaluation -- never at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The columns the model consumes. Kept explicit so train and inference always
# agree on feature order.
FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_21",
    "mom_63",
    "vol_21",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "sma_ratio_10_50",
    "price_vs_sma20",
    "bb_pos",
    "vol_change",
    "high_low_range",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the technical-indicator feature set for one ticker's history."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Trailing returns over several windows.
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_21"] = close.pct_change(21)
    out["mom_63"] = close.pct_change(63)            # ~quarter momentum

    # Realised volatility (annualised) of daily returns.
    out["vol_21"] = close.pct_change().rolling(21).std() * np.sqrt(252)

    out["rsi_14"] = _rsi(close, 14)

    macd, signal, hist = _macd(close)
    out["macd"] = macd / close                       # normalise by price
    out["macd_signal"] = signal / close
    out["macd_hist"] = hist / close

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["sma_ratio_10_50"] = sma10 / sma50 - 1
    out["price_vs_sma20"] = close / sma20 - 1

    # Position within the 20-day Bollinger band (0 = lower band, 1 = upper).
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_pos"] = (close - lower) / (upper - lower)

    out["vol_change"] = df["volume"].pct_change(5)
    out["high_low_range"] = (df["high"] - df["low"]) / close

    return out[FEATURE_COLUMNS]


def build_label(df: pd.DataFrame, horizon: int, threshold: float = 0.0) -> pd.Series:
    """Binary label: 1 if the forward `horizon`-day return exceeds `threshold`.

    Uses future prices, so rows near the end of history will be NaN (no future
    yet) and must be dropped before training.
    """
    close = df["close"]
    fwd_ret = close.shift(-horizon) / close - 1
    label = (fwd_ret > threshold).astype("float")
    label[fwd_ret.isna()] = np.nan
    return label.rename("label")


def build_forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """The raw forward return -- used by the backtest to score realised P&L."""
    close = df["close"]
    return (close.shift(-horizon) / close - 1).rename("fwd_ret")


def build_dataset(
    histories: dict[str, pd.DataFrame], horizon: int, threshold: float = 0.0
) -> pd.DataFrame:
    """Stack every ticker into one long, panel-style training table.

    Returns a DataFrame indexed by (date, ticker) with FEATURE_COLUMNS + label +
    fwd_ret. Rows with missing features or label are dropped.
    """
    frames = []
    for ticker, df in histories.items():
        feats = build_features(df)
        feats["label"] = build_label(df, horizon, threshold)
        feats["fwd_ret"] = build_forward_return(df, horizon)
        feats["ticker"] = ticker
        feats["date"] = df.index
        frames.append(feats)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["label"])
    panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel
