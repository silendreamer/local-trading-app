from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd

from trading_app.scanner2.config import MARKET_TIMEZONE, Scanner2Config
from trading_app.scanner2.output_builder import build_final_dataframe, output_columns
from trading_app.scanner2.polygon_client import PolygonRestClient
from trading_app.scanner2.scanner import (
    calculate_gap_pct,
    get_premarket_cumulative_volume,
    qualifies,
    run_full_scan,
)
from trading_app.scanner2.snapshot_store import save_snapshot


def test_calculate_gap_pct() -> None:
    assert calculate_gap_pct(12.0, 10.0) == 20.0


def test_qualifies_applies_momentum_rules() -> None:
    config = Scanner2Config(polygon_api_key="key")
    row = {
        "current_price": 12.0,
        "prev_close_4pm": 10.0,
        "prev_day_volume": 1_000_000,
        "premarket_volume": 600_000,
        "error": "",
    }

    assert qualifies(row, config) is True


def test_premarket_volume_caps_at_930() -> None:
    run_date = date(2026, 4, 30)
    index = pd.DatetimeIndex(
        [
            datetime.combine(run_date, time(8, 0), tzinfo=MARKET_TIMEZONE),
            datetime.combine(run_date, time(9, 29), tzinfo=MARKET_TIMEZONE),
            datetime.combine(run_date, time(9, 31), tzinfo=MARKET_TIMEZONE),
        ]
    )
    bars = pd.DataFrame({"volume": [100, 200, 300], "high": [1, 1, 1], "close": [1, 1, 1]}, index=index)

    volume = get_premarket_cumulative_volume(
        "AAA",
        datetime.combine(run_date, time(9, 45), tzinfo=MARKET_TIMEZONE),
        bars,
    )

    assert volume == 300


class FakePolygonClient:
    snapshot_calls = 0

    def get_grouped_daily_bars(self, target_date):
        return ok(
            {
                "results": [
                    {"T": "AAA", "c": 10.0, "v": 1_000_000},
                    {"T": "BBB", "c": 10.0, "v": 1_000_000},
                ]
            }
        )

    def get_all_tickers_snapshot(self):
        self.snapshot_calls += 1
        raise AssertionError("Scanner should load persisted snapshots instead of calling snapshot endpoint")

    def get_intraday_minute_bars(self, ticker, from_datetime, to_datetime):
        del from_datetime
        close = 13.0 if ticker == "AAA" else 9.0
        volume = 350_000 if ticker == "AAA" else 350_000
        timestamps = [
            datetime.combine(to_datetime.date(), time(8, 0), tzinfo=MARKET_TIMEZONE),
            datetime.combine(to_datetime.date(), time(9, 29), tzinfo=MARKET_TIMEZONE),
            to_datetime,
        ]
        return ok(
            {
                "results": [
                    {
                        "t": int(timestamp.timestamp() * 1000),
                        "o": close,
                        "h": close + 0.2,
                        "l": close - 0.2,
                        "c": close,
                        "v": volume,
                    }
                    for timestamp in timestamps
                    if timestamp <= to_datetime
                ]
            }
        )

    def get_latest_price(self, ticker):
        return ok({"results": {"p": 13.0 if ticker == "AAA" else 9.0}})


def ok(data):
    class Result:
        ok = True
        error = ""

        def __init__(self, payload):
            self.data = payload

    return Result(data)


def test_run_full_scan_narrows_candidates(tmp_path) -> None:
    config = Scanner2Config(
        polygon_api_key="key",
        min_gap_pct=20.0,
        min_prev_day_volume=500_000,
        min_premarket_volume=100_000,
        premarket_volume_to_prev_day_ratio=0.5,
        scan_times=("08:00", "08:30", "09:45"),
    )

    for scan_clock in [time(8, 0), time(8, 30), time(9, 45)]:
        save_snapshot(
            {
                "tickers": [
                    {
                        "ticker": "AAA",
                        "lastTrade": {"p": 13.0},
                        "day": {"v": 700_000},
                        "prevDay": {"c": 10.0, "v": 1_000_000},
                    },
                    {
                        "ticker": "BBB",
                        "lastTrade": {"p": 9.0},
                        "day": {"v": 700_000},
                        "prevDay": {"c": 10.0, "v": 1_000_000},
                    },
                ]
            },
            datetime.combine(date(2026, 4, 30), scan_clock, tzinfo=MARKET_TIMEZONE),
            snapshot_dir=tmp_path,
        )
    client = FakePolygonClient()

    result = run_full_scan(config=config, client=client, run_date=date(2026, 4, 30), snapshot_dir=tmp_path)

    qualified = result[result["still_qualified"]]

    assert list(qualified["ticker"]) == ["AAA"]
    assert qualified.iloc[0]["still_qualified"] == True
    assert qualified.iloc[0]["final_rank"] == 1
    assert client.snapshot_calls == 0


def test_build_final_dataframe_keeps_error_column() -> None:
    config = Scanner2Config(polygon_api_key="key")
    frame = build_final_dataframe(
        {"08_00": {"BAD": {"ticker": "BAD", "qualified": False, "error": "missing bars"}}},
        {},
        config,
    )

    assert frame.iloc[0]["ticker"] == "BAD"
    assert "missing bars" in frame.iloc[0]["error"]


def test_scanner2_output_columns_are_unique() -> None:
    columns = output_columns()

    assert len(columns) == len(set(columns))
    assert "premarket_volume_4am_to_9_30" in columns
    assert "final_premarket_volume_4am_to_9_30" in columns


def test_polygon_client_returns_structured_403() -> None:
    class FakeResponse:
        status_code = 403

        def raise_for_status(self):
            raise AssertionError("not called")

    class FakeSession:
        def get(self, url, params, timeout):
            return FakeResponse()

    client = PolygonRestClient("key", session=FakeSession(), request_sleep_seconds=0)

    result = client.get_all_tickers_snapshot()

    assert result.ok is False
    assert "403 Forbidden" in result.error
