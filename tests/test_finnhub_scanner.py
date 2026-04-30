from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_app.scanners.finnhub import (
    MARKET_TIMEZONE,
    FinnhubClient,
    FinnhubScannerConfig,
    RateLimiter,
    load_or_fetch_us_symbols,
    scan_finnhub_momentum,
)


NOW = datetime(2026, 4, 29, 10, 0, tzinfo=MARKET_TIMEZONE)


class FakeFinnhubClient:
    def __init__(self) -> None:
        self.symbol_calls = 0
        self.quote_calls = []
        self.candle_calls = []

    def stock_symbols(self, exchange="US"):
        self.symbol_calls += 1
        return [
            {"symbol": "AAA", "currency": "USD", "type": "Common Stock"},
            {"symbol": "BBB", "currency": "USD", "type": "Common Stock"},
            {"symbol": "FOREIGN", "currency": "CAD", "type": "Common Stock"},
            {"symbol": "BAD/SYM", "currency": "USD", "type": "Common Stock"},
        ]

    def quote(self, symbol):
        self.quote_calls.append(symbol)
        quotes = {
            "AAA": {"c": 10.6, "pc": 10.0, "t": int(NOW.timestamp())},
            "BBB": {"c": 10.4, "pc": 10.0, "t": int(NOW.timestamp())},
        }
        return quotes[symbol]

    def stock_candles(self, symbol, resolution, start_ts, end_ts):
        self.candle_calls.append((symbol, resolution))
        if resolution == "D":
            return {
                "s": "ok",
                "t": [int(pd.Timestamp("2026-04-28", tz=MARKET_TIMEZONE).timestamp())],
                "o": [9.5],
                "h": [10.2],
                "l": [9.0],
                "c": [10.0],
                "v": [1_000_000],
            }
        return {
            "s": "ok",
            "t": [
                int(pd.Timestamp("2026-04-29 04:00", tz=MARKET_TIMEZONE).timestamp()),
                int(pd.Timestamp("2026-04-29 09:29", tz=MARKET_TIMEZONE).timestamp()),
                int(pd.Timestamp("2026-04-29 09:30", tz=MARKET_TIMEZONE).timestamp()),
            ],
            "o": [10.1, 10.2, 10.5],
            "h": [10.3, 10.5, 10.7],
            "l": [10.0, 10.1, 10.4],
            "c": [10.2, 10.4, 10.6],
            "v": [75_000, 75_000, 20_000],
        }


def test_symbol_cache_is_reused_per_day(tmp_path) -> None:
    client = FakeFinnhubClient()

    first = load_or_fetch_us_symbols(client, cache_dir=tmp_path, now=NOW)
    second = load_or_fetch_us_symbols(client, cache_dir=tmp_path, now=NOW)

    assert first == ["AAA", "BBB"]
    assert second == ["AAA", "BBB"]
    assert client.symbol_calls == 1


def test_finnhub_scanner_prefilters_with_quotes_before_candles(tmp_path) -> None:
    client = FakeFinnhubClient()

    result = scan_finnhub_momentum(
        api_key="key",
        config=FinnhubScannerConfig(
            min_gap_pct=5.0,
            min_premarket_volume=100_000,
            min_price=2.0,
            max_price=20.0,
            top_n=10,
            requests_per_minute=0,
        ),
        now=NOW,
        cache_dir=tmp_path,
        client=client,
    )

    assert client.quote_calls == ["AAA", "BBB"]
    assert client.candle_calls == [("AAA", "1"), ("AAA", "D")]
    assert list(result["ticker"]) == ["AAA"]
    assert result.iloc[0]["pre_market_volume"] == 150_000
    assert result.iloc[0]["relative_volume"] == 0.15
    assert result.iloc[0]["continuation_signal"] == "BREAKOUT_CONTINUATION"


def test_rate_limiter_can_be_disabled() -> None:
    limiter = RateLimiter(0)

    limiter.wait()
    limiter.wait()

    assert limiter.min_interval == 0.0


def test_finnhub_client_adds_token_to_requests() -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"c": 1, "pc": 1}

    class FakeSession:
        def __init__(self):
            self.params = None

        def get(self, url, params, timeout):
            self.params = params
            return FakeResponse()

    session = FakeSession()
    client = FinnhubClient("secret-token", session=session, requests_per_minute=0)

    client.quote("AAPL")

    assert session.params["token"] == "secret-token"
    assert session.params["symbol"] == "AAPL"
