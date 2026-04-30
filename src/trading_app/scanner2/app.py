from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from trading_app.scanner2.config import MARKET_TIMEZONE, load_config
from trading_app.scanner2.scanner import run_full_scan


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    """Run scanner2 from the command line and save the watchlist CSV."""
    config = load_config()
    frame = run_full_scan(config=config)
    print(frame.to_string(index=False))
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"premarket_watchlist_{datetime.now(MARKET_TIMEZONE):%Y%m%d}.csv"
    frame.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
