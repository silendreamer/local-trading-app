from __future__ import annotations

import argparse
import logging

from trading_app.scanner2.config import load_config
from trading_app.scanner2.snapshot_store import run_snapshot_service


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    """Run the Scanner2 Polygon snapshot persistence service."""
    parser = argparse.ArgumentParser(description="Persist Polygon snapshots every N minutes.")
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--retention-days", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_snapshot_service(
        config=load_config(),
        interval_minutes=args.interval_minutes,
        poll_seconds=args.poll_seconds,
        retention_days=args.retention_days,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()
