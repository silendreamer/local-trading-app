from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppSettings:
    app_env: str = "local"
    log_level: str = "INFO"
    paper_starting_cash: float = 100_000.0
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    dry_run: bool = True


def load_environment(env_path: Path | None = None) -> AppSettings:
    """Load environment variables from .env and return typed settings."""
    import os

    load_dotenv(env_path or PROJECT_ROOT / ".env", override=True)
    return AppSettings(
        app_env=os.getenv("APP_ENV", "local"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        paper_starting_cash=float(os.getenv("PAPER_STARTING_CASH", "100000")),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        alpaca_paper=os.getenv("ALPACA_PAPER", "true").strip().lower() not in {"false", "0", "no"},
        dry_run=os.getenv("DRY_RUN", "true").strip().lower() not in {"false", "0", "no"},
    )


def load_tickers(path: Path | None = None) -> list[str]:
    """Load and validate tickers from config/tickers.yaml."""
    config_path = path or PROJECT_ROOT / "config" / "tickers.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"{config_path} must contain a non-empty 'tickers' list")

    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{config_path} contains duplicate tickers")
    if len(normalized) != 20:
        raise ValueError(f"{config_path} must contain exactly 20 tickers")

    return normalized
