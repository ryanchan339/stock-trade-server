"""Train the model on Yahoo Finance history and save the artifact.

Usage:  python -m scripts.train [--refresh]
"""
from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.data import download_history, download_universe
from src.features import build_dataset
from src.model import train_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the trading model.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of price data.")
    args = parser.parse_args()

    cfg = load_config()
    histories = download_universe(
        cfg["universe"],
        cfg["data"]["start"],
        end=None,
        cache_dir=cfg["data"]["cache_dir"],
        refresh=args.refresh,
    )

    benchmark = None
    if cfg["model"].get("relative_label"):
        benchmark = download_history(
            cfg["model"]["benchmark"],
            cfg["data"]["start"],
            cache_dir=cfg["data"]["cache_dir"],
            refresh=args.refresh,
        )

    panel = build_dataset(
        histories,
        horizon=cfg["model"]["horizon"],
        threshold=cfg["model"]["label_threshold"],
        benchmark=benchmark,
    )
    label_kind = "market-relative" if benchmark is not None else "absolute"
    logging.info(
        "Training panel: %d rows across %d tickers (%s labels)",
        len(panel), len(histories), label_kind,
    )

    model = train_model(
        panel,
        horizon=cfg["model"]["horizon"],
        threshold=cfg["model"]["label_threshold"],
        params=cfg["model"]["params"],
    )
    model.save(cfg["model"]["artifact"])


if __name__ == "__main__":
    main()
