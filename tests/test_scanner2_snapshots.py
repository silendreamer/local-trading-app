from __future__ import annotations

from datetime import datetime

from trading_app.scanner2.config import MARKET_TIMEZONE
from trading_app.scanner2.snapshot_store import (
    due_scan_time,
    load_snapshot,
    nearest_snapshot_time,
    save_snapshot,
    snapshot_exists,
    snapshot_path,
)


def test_snapshot_save_load_and_path(tmp_path) -> None:
    scan_time = datetime(2026, 4, 30, 8, 15, tzinfo=MARKET_TIMEZONE)
    payload = {"tickers": [{"ticker": "AAA"}]}

    path = save_snapshot(payload, scan_time, tmp_path)

    assert path == snapshot_path(scan_time, tmp_path)
    assert snapshot_exists(scan_time, tmp_path) is True
    assert load_snapshot(scan_time, tmp_path) == payload


def test_due_scan_time_floors_to_interval() -> None:
    now = datetime(2026, 4, 30, 8, 44, 59, tzinfo=MARKET_TIMEZONE)

    assert due_scan_time(now).strftime("%H:%M") == "08:30"


def test_nearest_snapshot_time_uses_latest_before_scan(tmp_path) -> None:
    early = datetime(2026, 4, 30, 8, 15, tzinfo=MARKET_TIMEZONE)
    later = datetime(2026, 4, 30, 8, 30, tzinfo=MARKET_TIMEZONE)
    save_snapshot({"tickers": []}, early, tmp_path)
    save_snapshot({"tickers": []}, later, tmp_path)

    nearest = nearest_snapshot_time(datetime(2026, 4, 30, 8, 40, tzinfo=MARKET_TIMEZONE), tmp_path)

    assert nearest == later
