"""Entry point for the daily live (paper) trading job.

Usage:
    python -m scripts.run_daily            # live: submits orders to Alpaca paper
    python -m scripts.run_daily --dry-run  # score + print targets, no orders

Schedule this on the server once per trading day, e.g. via cron at 09:35 ET:
    35 9 * * 1-5  cd /path/to/app && venv/bin/python -m scripts.run_daily >> logs/daily.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging

from src.config import load_config
from src.trade import run_daily


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the daily paper-trading rebalance.")
    parser.add_argument("--dry-run", action="store_true", help="Score and print targets without trading.")
    args = parser.parse_args()

    cfg = load_config()
    summary = run_daily(cfg, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
