"""Walk-forward backtest of the daily-rebalance strategy.

Design choices that keep the test honest:
  * The model is retrained on a rolling window and only ever sees labels whose
    forward window has fully closed (cutoff = decision_day - horizon), so no
    future information leaks into training.
  * A decision made using features at day t earns the market return from t to
    t+1. We never trade on information we wouldn't have had.
  * Turnover is charged a configurable round-trip cost in basis points.

The per-position stop-loss is a *live-trading* risk overlay and is intentionally
not modelled here; the backtest measures the raw signal + portfolio rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_dataset
from .model import train_model
from .strategy import StrategyParams, target_weights

log = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: dict[str, float]
    benchmark_curve: pd.Series | None = None

    def summary(self) -> str:
        lines = ["=== Backtest results ==="]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<22} {v:>10.4f}")
        return "\n".join(lines)


def _performance_metrics(daily: pd.Series, cost_note: str = "") -> dict[str, float]:
    daily = daily.dropna()
    if daily.empty:
        return {}
    total_return = (1 + daily).prod() - 1
    years = len(daily) / TRADING_DAYS
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    curve = (1 + daily).cumprod()
    drawdown = (curve / curve.cummax() - 1).min()
    win_rate = (daily > 0).mean()
    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate_daily": win_rate,
        "num_days": float(len(daily)),
    }


def run_backtest(
    histories: dict[str, pd.DataFrame],
    cfg: dict,
    benchmark: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the walk-forward simulation over the supplied price histories."""
    horizon = cfg["model"]["horizon"]
    threshold = cfg["model"]["label_threshold"]
    model_params = cfg["model"]["params"]
    bt = cfg["backtest"]
    sparams = StrategyParams.from_config(cfg)

    label_benchmark = benchmark if cfg["model"].get("relative_label") else None
    panel = build_dataset(histories, horizon, threshold, benchmark=label_benchmark)
    if panel.empty:
        raise ValueError("Empty feature panel -- not enough history.")

    # Daily simple returns matrix (dates x tickers) for P&L accrual.
    ret_matrix = pd.DataFrame(
        {t: df["close"].pct_change() for t, df in histories.items()}
    ).sort_index()

    all_dates = ret_matrix.index
    train_window = pd.Timedelta(days=int(bt["train_years"] * 365.25))
    retrain_every = bt["retrain_every"]
    cost_rate = bt["cost_bps"] / 1e4

    # Warm-up: need a full training window before the first decision.
    first_date = panel.index.get_level_values("date").min()
    start_decision = first_date + train_window
    decision_dates = all_dates[all_dates >= start_decision]
    if len(decision_dates) < 2:
        raise ValueError("Not enough history after the warm-up window to backtest.")

    log.info(
        "Backtesting %d decision days from %s to %s",
        len(decision_dates),
        decision_dates[0].date(),
        decision_dates[-1].date(),
    )

    panel_dates = panel.index.get_level_values("date")
    current_w = pd.Series(dtype=float)
    model = None
    days_since_train = retrain_every  # force a train on the first day
    strat_returns: dict[pd.Timestamp, float] = {}

    for i in range(len(decision_dates) - 1):
        t = decision_dates[i]
        t_next = decision_dates[i + 1]

        # Periodically retrain on the trailing window, using only labels whose
        # forward horizon has already resolved by day t.
        if days_since_train >= retrain_every:
            label_cutoff = t - pd.Timedelta(days=horizon * 2 + 1)
            train_mask = (panel_dates > t - train_window) & (panel_dates <= label_cutoff)
            train_slice = panel[train_mask]
            if len(train_slice) >= 500:
                model = train_model(train_slice, horizon, threshold, model_params).estimator
                days_since_train = 0
        days_since_train += 1

        # Score every ticker that has features on day t.
        target = pd.Series(dtype=float)
        if model is not None and t in panel_dates:
            today = panel.xs(t, level="date")
            probs = pd.Series(
                model.predict_proba(today[FEATURE_COLUMNS].to_numpy())[:, 1],
                index=today.index,
            )
            target = target_weights(probs, sparams)

        # Charge turnover cost on the move from current to target.
        all_names = target.index.union(current_w.index)
        turnover = (
            target.reindex(all_names, fill_value=0.0)
            - current_w.reindex(all_names, fill_value=0.0)
        ).abs().sum()
        cost = turnover * cost_rate
        current_w = target

        # Portfolio earns each held name's return from t to t_next.
        period_ret = ret_matrix.loc[t_next].reindex(current_w.index).fillna(0.0)
        gross = float((current_w * period_ret).sum())
        strat_returns[t_next] = gross - cost

    daily = pd.Series(strat_returns).sort_index()
    equity = bt["initial_cash"] * (1 + daily).cumprod()
    metrics = _performance_metrics(daily)

    benchmark_curve = None
    if benchmark is not None and not benchmark.empty:
        bench_ret = benchmark["close"].pct_change().reindex(daily.index).fillna(0.0)
        benchmark_curve = bt["initial_cash"] * (1 + bench_ret).cumprod()
        bench_metrics = _performance_metrics(bench_ret)
        for k, v in bench_metrics.items():
            metrics[f"benchmark_{k}"] = v

    return BacktestResult(
        equity_curve=equity,
        daily_returns=daily,
        metrics=metrics,
        benchmark_curve=benchmark_curve,
    )
