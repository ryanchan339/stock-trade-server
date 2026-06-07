"""Streamlit showcase dashboard for the Stock Trade Server.

Reads only the static files in results/ (committed by the daily GitHub Action),
so it has no dependency on Alpaca, scikit-learn, or yfinance. Deploy it for free
on Streamlit Community Cloud by pointing it at this repo / this file.

Local preview:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Stock Trade Server", page_icon="📈", layout="wide")


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


latest = load_json(RESULTS / "latest.json")
backtest = load_json(RESULTS / "backtest.json")

st.title("📈 Stock Trade Server")
st.caption(
    "A daily-rebalance machine-learning swing trader running on Alpaca **paper** "
    "trading. Educational project — not investment advice."
)
st.markdown(
    "[Source on GitHub](https://github.com/ryanchan339/stock-trade-server)"
)

today_tab, backtest_tab, about_tab = st.tabs(["Today", "Backtest", "How it works"])

# --------------------------------------------------------------------------- #
# Today
# --------------------------------------------------------------------------- #
with today_tab:
    if not latest:
        st.info("No live run recorded yet. The daily job will populate this.")
    else:
        st.subheader(f"Latest run — {latest.get('date', '')}")
        c = st.columns(4)
        eq = latest.get("equity_after")
        c[0].metric("Paper equity", f"${eq:,.0f}" if eq else "—")
        c[1].metric("Open positions", len(latest.get("targets", {})))
        c[2].metric("Market open", str(latest.get("market_open", "—")))
        stopped = latest.get("stopped_out", [])
        c[3].metric("Stopped out", len(stopped))

        targets = latest.get("targets", {})
        if targets:
            st.markdown("**Target portfolio (today)**")
            tdf = pd.DataFrame({"weight": targets}).sort_values("weight", ascending=False)
            st.bar_chart(tdf)
        else:
            st.write("Model is fully in cash today (nothing cleared the confidence threshold).")

        scores = latest.get("scores", {})
        if scores:
            st.markdown("**Model scores — P(beats market over next 5 days)**")
            sdf = (
                pd.DataFrame({"up_probability": scores})
                .sort_values("up_probability", ascending=False)
            )
            st.dataframe(sdf.style.format({"up_probability": "{:.1%}"}), use_container_width=True)

    log_path = RESULTS / "daily_log.csv"
    if log_path.exists():
        ldf = pd.read_csv(log_path)
        ldf = ldf.dropna(subset=["equity"])
        if not ldf.empty:
            st.markdown("**Paper account equity over time**")
            ldf["date"] = pd.to_datetime(ldf["date"])
            st.line_chart(ldf.set_index("date")["equity"])

# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
with backtest_tab:
    if not backtest:
        st.info("No backtest recorded yet. Run `python -m scripts.backtest`.")
    else:
        m = backtest["metrics"]
        st.subheader("Walk-forward backtest vs SPY buy & hold")
        c = st.columns(4)
        c[0].metric("Strategy CAGR", f"{m.get('cagr', 0) * 100:.1f}%",
                    f"{(m.get('cagr', 0) - m.get('benchmark_cagr', 0)) * 100:+.1f}% vs SPY")
        c[1].metric("Strategy Sharpe", f"{m.get('sharpe', 0):.2f}",
                    f"{m.get('sharpe', 0) - m.get('benchmark_sharpe', 0):+.2f} vs SPY")
        c[2].metric("Max drawdown", f"{m.get('max_drawdown', 0) * 100:.1f}%")
        c[3].metric("SPY CAGR", f"{m.get('benchmark_cagr', 0) * 100:.1f}%")

        eq = pd.Series(backtest["equity_curve"])
        eq.index = pd.to_datetime(eq.index)
        curves = pd.DataFrame({"Strategy": eq})
        if "benchmark_curve" in backtest:
            bench = pd.Series(backtest["benchmark_curve"])
            bench.index = pd.to_datetime(bench.index)
            curves["SPY buy & hold"] = bench
        st.line_chart(curves)
        st.caption(f"Backtest generated {backtest.get('generated', '')}.")

        with st.expander("All metrics"):
            st.dataframe(
                pd.Series(m, name="value").to_frame(), use_container_width=True
            )

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with about_tab:
    st.markdown(
        """
### How it works

```
Yahoo Finance daily prices
        │
        ▼
  technical features (RSI, MACD, momentum, moving-average ratios,
        │             volatility, Bollinger position, volume)
        ▼
  gradient-boosted classifier  →  P(stock beats SPY over next 5 days)
        │
        ▼
  portfolio rules  →  long the top-N highest-confidence names,
        │              confidence-weighted, per-name cap + stop-loss
        ▼
  Alpaca paper account  →  daily market-order rebalance
```

- **Daily swing horizon**, not true HFT — Yahoo data is daily and Alpaca is a
  REST API, so the signal-to-noise of a daily strategy is far better than
  chasing per-minute noise.
- **Market-relative training target.** The model predicts whether a stock will
  *beat the index*, not merely rise — so it's rewarded for stock selection, not
  for riding market beta you'd get free from SPY.
- **Walk-forward backtest** with rolling retraining and no lookahead.

The daily job runs on a free GitHub Actions cron; this dashboard is the static
showcase of its output.

---
*Educational software for paper trading only. Markets involve risk of loss.
Nothing here is financial advice.*
"""
    )
