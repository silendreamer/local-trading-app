from __future__ import annotations

from datetime import datetime

from trading_app.scanner2.config import MARKET_TIMEZONE
from trading_app.scanner2.snapshot_store import (
    delete_snapshots_older_than,
    due_scan_time,
    is_interval_boundary,
    load_snapshot,
    nearest_snapshot_time,
    recent_snapshots,
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


def test_interval_boundary_matches_quarter_hour_clock_times() -> None:
    boundary = datetime(2026, 4, 30, 13, 15, 35, tzinfo=MARKET_TIMEZONE)
    between_boundaries = datetime(2026, 4, 30, 13, 7, 0, tzinfo=MARKET_TIMEZONE)

    assert is_interval_boundary(boundary) is True
    assert is_interval_boundary(between_boundaries) is False


def test_delete_snapshots_older_than_retention_window(tmp_path) -> None:
    now = datetime(2026, 4, 30, 13, 0, tzinfo=MARKET_TIMEZONE)
    old_snapshot_time = datetime(2026, 4, 26, 13, 0, tzinfo=MARKET_TIMEZONE)
    recent_snapshot_time = datetime(2026, 4, 28, 13, 0, tzinfo=MARKET_TIMEZONE)
    old_path = save_snapshot({"tickers": []}, old_snapshot_time, tmp_path)
    recent_path = save_snapshot({"tickers": []}, recent_snapshot_time, tmp_path)

    deleted = delete_snapshots_older_than(3, tmp_path, now=now)

    assert deleted == 1
    assert old_path.exists() is False
    assert recent_path.exists() is True


def test_recent_snapshots_returns_latest_five_by_capture_time(tmp_path) -> None:
    times = [
        datetime(2026, 4, 30, 8, minute, tzinfo=MARKET_TIMEZONE)
        for minute in [0, 15, 30, 45]
    ] + [
        datetime(2026, 4, 30, 9, minute, tzinfo=MARKET_TIMEZONE)
        for minute in [0, 15]
    ]
    for scan_time in times:
        save_snapshot({"tickers": []}, scan_time, tmp_path)

    recent = recent_snapshots(5, tmp_path)

    assert [scan_time.strftime("%H:%M") for scan_time, _ in recent] == [
        "09:15",
        "09:00",
        "08:45",
        "08:30",
        "08:15",
    ]


def test_nearest_snapshot_time_uses_latest_before_scan(tmp_path) -> None:
    early = datetime(2026, 4, 30, 8, 15, tzinfo=MARKET_TIMEZONE)
    later = datetime(2026, 4, 30, 8, 30, tzinfo=MARKET_TIMEZONE)
    save_snapshot({"tickers": []}, early, tmp_path)
    save_snapshot({"tickers": []}, later, tmp_path)

    nearest = nearest_snapshot_time(datetime(2026, 4, 30, 8, 40, tzinfo=MARKET_TIMEZONE), tmp_path)

    assert nearest == later
