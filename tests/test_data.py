from __future__ import annotations

import pandas as pd

from trading_app.data import (
    MarketDataRequest,
    fetch_ohlc,
    parse_polygon_ohlc,
    parse_polygon_snapshot_quote,
    polygon_interval_parts,
)


def polygon_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def polygon_ns(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1_000_000_000)


def test_parse_polygon_daily_ohlc_returns_price_frame() -> None:
    payload = {
        "status": "OK",
        "results": [
            {"t": polygon_ms("2024-01-01"), "o": 99.0, "h": 102.0, "l": 98.0, "c": 100.0, "v": 900000},
            {"t": polygon_ms("2024-01-02"), "o": 100.0, "h": 103.0, "l": 99.0, "c": 101.5, "v": 1000000},
        ],
    }

    frame = parse_polygon_ohlc("AAA", payload)

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.iloc[-1]["Close"] == 101.5
    assert str(frame.index[0].date()) == "2024-01-01"


def test_parse_polygon_intraday_ohlc_returns_price_frame() -> None:
    payload = {
        "status": "OK",
        "results": [
            {"t": polygon_ms("2024-01-02 14:30"), "o": 99.0, "h": 102.0, "l": 98.0, "c": 100.0, "v": 900000},
            {"t": polygon_ms("2024-01-02 14:45"), "o": 100.0, "h": 103.0, "l": 99.0, "c": 101.5, "v": 1000000},
        ],
    }

    frame = parse_polygon_ohlc("AAA", payload, "15m")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.iloc[-1]["Close"] == 101.5
    assert str(frame.index[-1]) == "2024-01-02 09:45:00-05:00"


def test_polygon_interval_parts_uses_intraday_range_endpoint_parts() -> None:
    assert polygon_interval_parts("15m") == (15, "minute")
    assert polygon_interval_parts("1d") == (1, "day")


def test_parse_polygon_snapshot_quote_returns_latest_price() -> None:
    payload = {
        "status": "OK",
        "ticker": {
            "ticker": "AAPL",
            "lastTrade": {"p": 210.15, "t": polygon_ns("2026-04-27 15:55")},
            "prevDay": {"c": 209.0},
            "todaysChange": 1.15,
            "todaysChangePerc": 0.5502,
        },
    }

    quote = parse_polygon_snapshot_quote("AAPL", payload)

    assert quote.ticker == "AAPL"
    assert quote.price == 210.15
    assert quote.latest_trading_day == "2026-04-27"
    assert quote.previous_close == 209.0
    assert quote.change_percent == "0.5502%"


def test_fetch_ohlc_uses_polygon_request(monkeypatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "OK",
                "results": [
                    {
                        "t": polygon_ms("2026-04-27 13:45"),
                        "o": 100.0,
                        "h": 102.0,
                        "l": 99.0,
                        "c": 101.0,
                        "v": 1000,
                    }
                ],
            }

    def fake_get(url, params, timeout):
        captured.update({"url": url, **params})
        return FakeResponse()

    monkeypatch.setattr("trading_app.data.requests.get", fake_get)

    result = fetch_ohlc(MarketDataRequest(tickers=["AAPL"], start="2026-04-27", end="2026-04-28", interval="15m"))

    assert "AAPL" in result
    assert "/v2/aggs/ticker/AAPL/range/15/minute/2026-04-27/2026-04-28" in captured["url"]
    assert captured["adjusted"] == "true"
    assert captured["apiKey"] == "test-key"
