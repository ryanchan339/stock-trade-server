"""Walk-forward backtest with an SPY buy-and-hold benchmark.

Usage:  python -m scripts.backtest [--refresh] [--plot]
"""
from __future__ import annotations

import argparse
import logging

from src.backtest import run_backtest
from src.config import load_config
from src.data import download_history, download_universe
from src.results import save_backtest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backtest the strategy.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of price data.")
    parser.add_argument("--plot", action="store_true", help="Save an equity-curve PNG.")
    args = parser.parse_args()

    cfg = load_config()
    histories = download_universe(
        cfg["universe"],
        cfg["data"]["start"],
        end=None,
        cache_dir=cfg["data"]["cache_dir"],
        refresh=args.refresh,
    )
    benchmark = download_history(
        "SPY", cfg["data"]["start"], cache_dir=cfg["data"]["cache_dir"], refresh=args.refresh
    )

    result = run_backtest(histories, cfg, benchmark=benchmark)
    print(result.summary())
    save_backtest(result)
    print("Saved results/backtest.json")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            ax = result.equity_curve.plot(label="Strategy", figsize=(11, 6))
            if result.benchmark_curve is not None:
                result.benchmark_curve.plot(ax=ax, label="SPY buy & hold", alpha=0.7)
            ax.set_title("Strategy vs SPY")
            ax.set_ylabel("Equity ($)")
            ax.legend()
            plt.tight_layout()
            plt.savefig("artifacts/equity_curve.png", dpi=120)
            print("Saved artifacts/equity_curve.png")
        except ImportError:
            print("matplotlib not installed; skipping plot (pip install matplotlib).")


if __name__ == "__main__":
    main()
