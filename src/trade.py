"""Daily live-trading routine against Alpaca paper trading.

Run once per day (e.g. shortly after the market opens). It:
  1. Refreshes price history and rebuilds today's features.
  2. Loads the trained model and scores the universe.
  3. Builds target weights with the strategy rules.
  4. Applies stop-losses, then rebalances the Alpaca account toward the targets.

The model is trained offline by scripts/train.py; this routine only does
inference + execution, so the daily job is fast.
"""
from __future__ import annotations

import logging

import pandas as pd

from .broker import AlpacaBroker
from .config import AlpacaCredentials, load_config
from .data import download_universe
from .features import FEATURE_COLUMNS, build_features
from .model import TradeModel
from .strategy import StrategyParams, target_weights

log = logging.getLogger(__name__)


def score_universe(
    histories: dict[str, pd.DataFrame], model: TradeModel
) -> pd.Series:
    """Latest-day up-move probability for each ticker."""
    scores = {}
    for ticker, df in histories.items():
        feats = build_features(df).dropna(subset=FEATURE_COLUMNS)
        if feats.empty:
            continue
        latest = feats.iloc[[-1]]
        scores[ticker] = float(model.predict_proba(latest)[0])
    return pd.Series(scores).sort_values(ascending=False)


def run_daily(
    cfg: dict | None = None,
    creds: AlpacaCredentials | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute one daily rebalance. Returns a summary dict for logging."""
    cfg = cfg or load_config()
    sparams = StrategyParams.from_config(cfg)

    histories = download_universe(
        cfg["universe"],
        cfg["data"]["start"],
        end=None,
        cache_dir=cfg["data"]["cache_dir"],
    )
    model = TradeModel.load(cfg["model"]["artifact"])

    probs = score_universe(histories, model)
    targets = target_weights(probs, sparams)
    log.info("Top scores:\n%s", probs.head(8).to_string())
    log.info("Target weights:\n%s", targets.to_string() if not targets.empty else "  (all cash)")

    summary = {
        "scores": probs.round(4).to_dict(),
        "targets": targets.round(4).to_dict(),
        "stopped_out": [],
        "dry_run": dry_run,
    }

    if dry_run:
        log.info("Dry run -- no orders submitted.")
        return summary

    if creds is None:
        from .config import load_credentials

        creds = load_credentials(paper=cfg["broker"]["paper"])

    broker = AlpacaBroker(creds)
    if not broker.is_market_open():
        log.warning("Market is closed; skipping execution. Orders would queue otherwise.")
        summary["market_open"] = False
        return summary
    summary["market_open"] = True

    stopped = broker.apply_stop_losses(sparams.stop_loss)
    if stopped:
        log.info("Stopped out: %s", stopped)
    summary["stopped_out"] = stopped

    # Don't re-buy names we just stopped out of today.
    if stopped:
        targets = targets.drop(index=[s for s in stopped if s in targets.index])

    broker.rebalance_to(targets, sparams.rebalance_band)
    summary["equity_after"] = broker.equity()
    return summary
