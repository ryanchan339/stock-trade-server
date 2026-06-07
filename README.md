# Stock Trade Server

A daily-rebalance, machine-learning swing-trading bot for **Alpaca paper trading**.
It trains a gradient-boosted classifier on free Yahoo Finance history, scores a
universe of liquid large-caps once a day, and holds a small long-only portfolio
of the highest-confidence names — rebalancing daily with hard risk limits.

> **This is a research / educational project on a paper account. It is not
> investment advice and is not wired to real money.** See *Honest results* below:
> as configured it does **not** beat SPY buy-and-hold. Treat it as a framework to
> iterate on, not a money printer.

---

## How it works

```
Yahoo Finance (daily OHLCV)
        │
        ▼
  feature engineering ──►  technical indicators (RSI, MACD, momentum,
        │                  moving-average ratios, volatility, Bollinger…)
        ▼
  HistGradientBoosting  ──►  P(forward 5-day return > 0) per stock-day
        │
        ▼
  portfolio rules       ──►  long top-N names above a probability threshold,
        │                    confidence-weighted, per-name cap, stop-loss
        ▼
  Alpaca paper account  ──►  daily market-order rebalance toward target weights
```

### Why these choices
- **Daily horizon, not true HFT.** Yahoo data is daily and Alpaca is a
  REST API over the internet — microsecond HFT is impossible here. Daily swing
  trading has a cleaner signal-to-noise ratio and near-zero sensitivity to
  latency. (We discussed this trade-off before building.)
- **Gradient-boosted trees (`HistGradientBoostingClassifier`).** Fast, robust on
  tabular/technical features, hard to overfit with regularization, and ships with
  scikit-learn (no native build headaches). It predicts a *probability*, which we
  use directly for position sizing.
- **Walk-forward backtest.** The model is retrained on a rolling window and only
  ever trains on labels whose forward window has already resolved, so reported
  performance has no lookahead bias.

---

## Project layout

```
config.yaml            # universe, model, strategy, backtest, risk settings
.env.example           # Alpaca paper API keys (copy to .env)
src/
  config.py            # config + credential loading
  data.py              # yfinance download + local pickle cache
  features.py          # technical indicators + label construction
  model.py             # train / save / load the classifier
  strategy.py          # probabilities -> target portfolio weights
  backtest.py          # walk-forward simulation + metrics
  broker.py            # Alpaca trading wrapper (notional orders, stop-loss)
  trade.py             # the daily inference + execution routine
scripts/
  train.py             # python -m scripts.train
  backtest.py          # python -m scripts.backtest [--plot]
  run_daily.py         # python -m scripts.run_daily [--dry-run]
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Alpaca paper keys (https://app.alpaca.markets -> Paper Trading -> API Keys)
cp .env.example .env
# edit .env and paste your ALPACA_API_KEY / ALPACA_SECRET_KEY
```

## Usage

```bash
# 1. Train on Yahoo history (writes artifacts/model.joblib)
python -m scripts.train

# 2. Backtest with an SPY benchmark (add --plot for a PNG equity curve)
python -m scripts.backtest

# 3. See today's target portfolio WITHOUT trading
python -m scripts.run_daily --dry-run

# 4. Run the live paper-trading rebalance (needs .env keys)
python -m scripts.run_daily
```

### Running on a server

Schedule `run_daily` once per trading day, shortly after the open. Example cron
(09:35 US/Eastern, Mon–Fri):

```cron
35 9 * * 1-5  cd /path/to/Stock\ Trade\ Server && \
  venv/bin/python -m scripts.run_daily >> logs/daily.log 2>&1
```

Retrain weekly or monthly:

```cron
0 18 * * 0  cd /path/to/Stock\ Trade\ Server && venv/bin/python -m scripts.train
```

The job is safe to run when the market is closed — it logs a warning and skips
execution rather than queuing orders.

---

## Configuration

Everything tunable lives in `config.yaml`:

| Section     | Key                    | Meaning |
|-------------|------------------------|---------|
| `model`     | `horizon`              | Forward-return window the label is built on (days). |
| `model`     | `params`               | Gradient-boosting hyperparameters. |
| `strategy`  | `prob_threshold`       | Minimum up-probability to take a position. |
| `strategy`  | `max_positions`        | Max simultaneous long names. |
| `strategy`  | `max_weight_per_name`  | Per-name weight cap (diversification). |
| `strategy`  | `stop_loss`            | Exit a name down more than this from entry. |
| `backtest`  | `train_years`          | Rolling training window length. |
| `backtest`  | `retrain_every`        | Retrain cadence in trading days. |
| `backtest`  | `cost_bps`             | Assumed round-trip transaction cost. |

---

## Honest results

A walk-forward backtest over ~7 years of the default 15-stock universe (run it
yourself with `python -m scripts.backtest`):

| Metric            | Strategy | SPY buy & hold |
|-------------------|----------|----------------|
| CAGR              | ~9%      | ~16%           |
| Sharpe            | ~0.48    | ~0.84          |
| Max drawdown      | ~-41%    | ~-34%          |

**The model has real predictive signal** (out-of-sample it ranks up-moves better
than the base rate), but as configured the long-only strategy **underperforms
simply holding the index.** That is the normal, expected outcome — beating
buy-and-hold during a historic mega-cap bull market with a long-only book is
genuinely hard, and most naïve strategies lose to it. The value here is the
end-to-end framework you can now iterate on.

### Ideas to improve it
- **Market-relative labels:** predict whether a stock beats SPY, not whether it
  rises, to strip out the market beta the index already gives you for free.
- **Long/short:** short the lowest-probability names to hedge market risk.
- **Volatility targeting / regime filter:** cut exposure in high-volatility or
  downtrending regimes to tame the drawdown.
- **Bigger / sector-diversified universe** and proper hyperparameter search with
  purged, embargoed cross-validation.
- **Probability calibration** (`CalibratedClassifierCV`) so weights reflect true
  edge.

---

## Disclaimer

Educational software for **paper trading only**. Markets involve risk of loss.
Nothing here is financial advice. Do not point this at a live brokerage account
without understanding and accepting the consequences.
