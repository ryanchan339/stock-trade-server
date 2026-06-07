"""Portfolio construction: turn per-stock probabilities into target weights.

The strategy is long-only and confidence-weighted:
  1. Score every ticker with the model.
  2. Keep names whose up-move probability clears `prob_threshold`.
  3. Take the top `max_positions` by probability.
  4. Weight them proportionally to (prob - 0.5), capped per name, scaled so the
     book is `target_invested` of equity (the rest stays in cash).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyParams:
    prob_threshold: float = 0.55
    max_positions: int = 5
    max_weight_per_name: float = 0.25
    target_invested: float = 0.95
    stop_loss: float = 0.07
    rebalance_band: float = 0.03

    @classmethod
    def from_config(cls, cfg: dict) -> "StrategyParams":
        s = cfg["strategy"]
        return cls(
            prob_threshold=s["prob_threshold"],
            max_positions=s["max_positions"],
            max_weight_per_name=s["max_weight_per_name"],
            target_invested=s["target_invested"],
            stop_loss=s["stop_loss"],
            rebalance_band=s["rebalance_band"],
        )


def target_weights(probs: pd.Series, params: StrategyParams) -> pd.Series:
    """Map a Series of {ticker -> up-probability} to target portfolio weights.

    Returns weights that sum to <= target_invested. Empty Series if nothing
    qualifies.
    """
    qualified = probs[probs >= params.prob_threshold].sort_values(ascending=False)
    qualified = qualified.head(params.max_positions)
    if qualified.empty:
        return pd.Series(dtype=float)

    # Edge over a coin flip drives the raw allocation.
    edge = (qualified - 0.5).clip(lower=0)
    if edge.sum() == 0:
        weights = pd.Series(params.target_invested / len(qualified), index=qualified.index)
    else:
        weights = edge / edge.sum() * params.target_invested

    # Apply the per-name cap, then renormalise the leftover across the others.
    weights = _apply_cap(weights, params.max_weight_per_name, params.target_invested)
    return weights


def _apply_cap(weights: pd.Series, cap: float, budget: float) -> pd.Series:
    """Iteratively cap weights at `cap` and redistribute the excess."""
    w = weights.copy()
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any():
            break
        room = (cap - w[under])
        if room.sum() <= 0:
            break
        w[under] += room / room.sum() * min(excess, room.sum())
    return w.clip(upper=cap)


def diff_to_trades(
    target: pd.Series, current: pd.Series, band: float
) -> pd.Series:
    """Weight changes to execute, ignoring moves smaller than `band`.

    Positive = buy more, negative = sell. Names in `current` but not `target`
    are fully closed.
    """
    all_names = target.index.union(current.index)
    tgt = target.reindex(all_names, fill_value=0.0)
    cur = current.reindex(all_names, fill_value=0.0)
    delta = tgt - cur
    # Always allow full exits; otherwise apply the no-trade band.
    keep = (delta.abs() >= band) | (tgt == 0.0)
    return delta[keep]
