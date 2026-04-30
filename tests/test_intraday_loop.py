from __future__ import annotations

from datetime import datetime
import logging

from trading_app.intraday_loop import (
    MARKET_TIMEZONE,
    TradingLoopManager,
    format_market_time,
    is_market_open,
    next_market_open,
)


def test_market_open_uses_timezone_aware_new_york_hours() -> None:
    during_market = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)
    before_market = datetime(2026, 4, 27, 9, 0, tzinfo=MARKET_TIMEZONE)
    after_close = datetime(2026, 4, 27, 16, 0, tzinfo=MARKET_TIMEZONE)

    assert is_market_open(during_market)
    assert not is_market_open(before_market)
    assert not is_market_open(after_close)


def test_next_market_open_skips_weekends() -> None:
    friday_after_close = datetime(2026, 5, 1, 17, 0, tzinfo=MARKET_TIMEZONE)

    next_open = next_market_open(friday_after_close)

    assert next_open.weekday() == 0
    assert next_open.hour == 9
    assert next_open.minute == 30


def test_loop_does_not_start_outside_market_hours() -> None:
    manager = TradingLoopManager()
    after_close = datetime(2026, 4, 27, 16, 1, tzinfo=MARKET_TIMEZONE)

    message = manager.start(
        ["AAPL"],
        dry_run=True,
        alpaca_paper=True,
        now=after_close,
    )
    snapshot = manager.snapshot()

    assert not snapshot.running
    assert snapshot.status == "stopped"
    assert "market is closed" in message


def test_loop_logs_rejected_start_outside_market_hours(caplog) -> None:
    manager = TradingLoopManager()
    after_close = datetime(2026, 4, 27, 16, 1, tzinfo=MARKET_TIMEZONE)

    with caplog.at_level(logging.INFO, logger="trading_app.scheduler"):
        manager.start(["AAPL"], dry_run=True, alpaca_paper=True, now=after_close)

    assert "Scheduler start rejected outside market hours" in caplog.text


def test_loop_prevents_duplicate_scheduler_start_during_startup() -> None:
    manager = TradingLoopManager()
    market_time = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)
    manager._state.running = True
    manager._thread = None

    message = manager.start(["AAPL"], dry_run=True, alpaca_paper=True, now=market_time)

    assert message == "Trading loop is already running."


def test_format_market_time_includes_timezone() -> None:
    value = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)

    assert format_market_time(value).endswith("EDT")
