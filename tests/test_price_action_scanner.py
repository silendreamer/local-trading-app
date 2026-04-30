from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_app.data import MarketDataRequest
from trading_app.intraday_loop import MARKET_TIMEZONE
from trading_app.scanners.price_action import scan_price_action


def make_intraday_frame(values: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "High": [row[1] for row in values],
            "Low": [row[2] for row in values],
            "Close": [row[3] for row in values],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp(row[0], tz=MARKET_TIMEZONE) for row in values],
        ),
    )


def test_scanner_returns_buy_watch_for_fresh_breakout() -> None:
    now = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)
    frame = make_intraday_frame(
        [
            ("2026-04-27 09:00", 100.0, 98.0, 99.0),
            ("2026-04-27 09:15", 101.0, 99.0, 100.0),
            ("2026-04-27 09:30", 102.0, 100.0, 101.0),
            ("2026-04-27 09:45", 104.0, 101.0, 103.0),
        ]
    )

    def fetcher(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
        assert request.interval == "15m"
        return {"AAPL": frame}

    result = scan_price_action(["AAPL"], now=now, fetcher=fetcher)

    assert result[0].ticker == "AAPL"
    assert result[0].current_price == 103.0
    assert result[0].signal == "BUY_WATCH"
    assert result[0].data_freshness_status == "fresh"


def test_scanner_skips_when_market_is_closed() -> None:
    now = datetime(2026, 4, 27, 16, 1, tzinfo=MARKET_TIMEZONE)

    result = scan_price_action(["AAPL"], now=now, fetcher=lambda request: {})

    assert result[0].signal == "SKIP"
    assert result[0].data_freshness_status == "market_closed"
    assert result[0].reason == "Market is closed"


def test_scanner_skips_stale_intraday_data() -> None:
    now = datetime(2026, 4, 27, 11, 0, tzinfo=MARKET_TIMEZONE)
    frame = make_intraday_frame(
        [
            ("2026-04-27 09:00", 100.0, 98.0, 99.0),
            ("2026-04-27 09:15", 101.0, 99.0, 100.0),
            ("2026-04-27 09:30", 102.0, 100.0, 101.0),
            ("2026-04-27 09:45", 104.0, 101.0, 103.0),
        ]
    )

    result = scan_price_action(["AAPL"], now=now, fetcher=lambda request: {"AAPL": frame})

    assert result[0].signal == "SKIP"
    assert result[0].data_freshness_status == "stale"
    assert "stale" in result[0].reason


def test_scanner_skips_missing_ticker_data() -> None:
    now = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)

    result = scan_price_action(["AAPL"], now=now, fetcher=lambda request: {})

    assert result[0].signal == "SKIP"
    assert result[0].data_freshness_status == "missing"
    assert result[0].reason == "Missing intraday candles"


def test_scanner_reports_intraday_fetch_failure_without_daily_fallback_message() -> None:
    now = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)

    def failing_fetcher(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
        assert request.interval == "15m"
        raise ValueError("No market data returned")

    result = scan_price_action(["AAPL"], now=now, fetcher=failing_fetcher)

    assert result[0].signal == "SKIP"
    assert result[0].data_freshness_status == "missing"
    assert result[0].reason == "Market data fetch failed: No market data returned"
    assert "Yahoo chart fallback currently supports only 1d interval" not in result[0].reason
