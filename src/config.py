"""Load configuration from config.yaml and Alpaca credentials from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml into a plain dict."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str
    paper: bool = True


def load_credentials(paper: bool = True) -> AlpacaCredentials:
    """Pull Alpaca keys from environment / .env. Raises if missing."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Copy .env.example to .env and fill in "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY (paper trading keys)."
        )
    return AlpacaCredentials(api_key=api_key, secret_key=secret_key, paper=paper)
