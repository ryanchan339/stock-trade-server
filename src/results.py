"""Persist run outputs to results/ so the dashboard (and git history) can show them.

These files are committed back to the repo by the GitHub Actions job, so the
Streamlit dashboard only needs to read static JSON/CSV -- it never imports the
trading stack or touches Alpaca.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ROOT

RESULTS_DIR = ROOT / "results"
LATEST = RESULTS_DIR / "latest.json"
DAILY_LOG = RESULTS_DIR / "daily_log.csv"
BACKTEST = RESULTS_DIR / "backtest.json"


def _ensure() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_daily(summary: dict) -> dict:
    """Write the latest run summary and append one row to the daily log."""
    _ensure()
    summary = {**summary, "date": date.today().isoformat()}
    LATEST.write_text(json.dumps(summary, indent=2, default=str))

    row = {
        "date": summary["date"],
        "equity": summary.get("equity_after"),
        "num_positions": len(summary.get("targets", {})),
        "invested": ";".join(summary.get("targets", {}).keys()),
        "market_open": summary.get("market_open"),
        "stopped_out": ";".join(summary.get("stopped_out", [])),
    }
    if DAILY_LOG.exists():
        log = pd.read_csv(DAILY_LOG)
        log = log[log["date"] != row["date"]]  # idempotent for same-day reruns
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.sort_values("date").to_csv(DAILY_LOG, index=False)
    return summary


def save_backtest(result) -> None:
    """Persist backtest metrics + equity curves for the dashboard."""
    _ensure()
    payload = {
        "generated": date.today().isoformat(),
        "metrics": {k: float(v) for k, v in result.metrics.items()},
        "equity_curve": {
            d.strftime("%Y-%m-%d"): float(v) for d, v in result.equity_curve.items()
        },
    }
    if result.benchmark_curve is not None:
        payload["benchmark_curve"] = {
            d.strftime("%Y-%m-%d"): float(v) for d, v in result.benchmark_curve.items()
        }
    BACKTEST.write_text(json.dumps(payload))
