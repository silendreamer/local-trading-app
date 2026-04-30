from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_app.scanners.premarket import (
    MARKET_TIMEZONE,
    MomentumScannerConfig,
    normalize_tickers,
    scan_momentum_candidates,
    scan_premarket,
)


def daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "High": [99.0, 100.0, 105.0],
            "Close": [98.0, 99.0, 101.0],
            "Volume": [900_000, 1_000_000, 1_100_000],
        },
        index=pd.to_datetime(["2026-04-24", "2026-04-27", "2026-04-28"]),
    )


def intraday_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-04-28 04:00", tz=MARKET_TIMEZONE),
            pd.Timestamp("2026-04-28 04:05", tz=MARKET_TIMEZONE),
            pd.Timestamp("2026-04-28 09:29", tz=MARKET_TIMEZONE),
            pd.Timestamp("2026-04-28 09:30", tz=MARKET_TIMEZONE),
            pd.Timestamp("2026-04-28 10:00", tz=MARKET_TIMEZONE),
        ]
    )
    return pd.DataFrame(
        {
            "High": [102.0, 104.0, 103.0, 106.0, 107.0],
            "Low": [101.0, 102.0, 102.0, 104.0, 105.0],
            "Close": [101.5, 103.5, 102.5, 105.0, 106.0],
            "Volume": [100, 200, 300, 400, 500],
        },
        index=index,
    )


def test_normalize_tickers_accepts_lines_and_commas() -> None:
    assert normalize_tickers("aapl, msft\nAAPL") == ["AAPL", "MSFT"]


def test_scan_premarket_collects_expected_metrics() -> None:
    now = datetime(2026, 4, 28, 10, 1, tzinfo=MARKET_TIMEZONE)

    result = scan_premarket(
        ["AAPL"],
        now=now,
        daily_fetcher=lambda ticker: daily_frame(),
        intraday_fetcher=lambda ticker: intraday_frame(),
    )

    row = result.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["previous_day_high"] == 100.0
    assert row["previous_day_close"] == 99.0
    assert row["previous_day_volume"] == 1_000_000
    assert row["pre_market_high"] == 104.0
    assert row["pre_market_volume"] == 600
    assert row["current_price"] == 106.0
    assert round(row["percent_gap_from_previous_close"], 2) == 7.07
    assert round(row["percent_move_from_previous_day_high"], 2) == 6.0
    assert row["above_pre_market_high"] == True
    assert row["error"] == ""


def test_scan_premarket_returns_row_error_for_invalid_ticker() -> None:
    now = datetime(2026, 4, 28, 10, 1, tzinfo=MARKET_TIMEZONE)

    result = scan_premarket(
        ["BAD"],
        now=now,
        daily_fetcher=lambda ticker: pd.DataFrame(),
        intraday_fetcher=lambda ticker: intraday_frame(),
    )

    assert result.iloc[0]["ticker"] == "BAD"
    assert result.iloc[0]["error"] == "No daily history returned"


def test_scan_premarket_handles_empty_premarket_data() -> None:
    now = datetime(2026, 4, 28, 10, 1, tzinfo=MARKET_TIMEZONE)
    regular_only = intraday_frame().between_time("09:30", "16:00")

    result = scan_premarket(
        ["AAPL"],
        now=now,
        daily_fetcher=lambda ticker: daily_frame(),
        intraday_fetcher=lambda ticker: regular_only,
    )

    row = result.iloc[0]
    assert pd.isna(row["pre_market_high"])
    assert pd.isna(row["pre_market_volume"])
    assert row["current_price"] == 106.0
    assert row["above_pre_market_high"] is None
    assert row["error"] == "No premarket candles for today"


def test_scan_momentum_candidates_filters_and_sorts_top_names() -> None:
    now = datetime(2026, 4, 28, 10, 1, tzinfo=MARKET_TIMEZONE)

    def daily_fetcher(tickers):
        return {
            ticker: daily_frame()
            for ticker in tickers
        }

    def intraday_fetcher(tickers):
        frames = {}
        for ticker in tickers:
            frame = intraday_frame().copy()
            if ticker == "AAA":
                frame.loc[frame.index[-1], "Close"] = 110.0
                frame.loc[frame.index[:3], "Volume"] = [100_000, 100_000, 100_000]
            elif ticker == "BBB":
                frame.loc[frame.index[-1], "Close"] = 108.0
                frame.loc[frame.index[:3], "Volume"] = [200_000, 200_000, 200_000]
            else:
                frame.loc[frame.index[-1], "Close"] = 101.0
                frame.loc[frame.index[:3], "Volume"] = [1_000, 1_000, 1_000]
            frames[ticker] = frame
        return frames

    result = scan_momentum_candidates(
        ["AAA", "BBB", "CCC"],
        config=MomentumScannerConfig(
            gap_threshold_percent=5.0,
            min_premarket_volume=100_000,
            min_price=2.0,
            max_price=120.0,
            top_n=2,
        ),
        now=now,
        daily_fetcher=daily_fetcher,
        intraday_fetcher=intraday_fetcher,
    )

    assert list(result["ticker"]) == ["AAA", "BBB"]
    assert result.iloc[0]["continuation_signal"] == "BREAKOUT_CONTINUATION"
    assert result.iloc[0]["relative_volume"] == 0.3
