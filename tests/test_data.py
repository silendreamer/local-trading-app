from __future__ import annotations

import pandas as pd

from trading_app.data import (
    MarketDataRequest,
    alpha_vantage_params,
    fetch_ohlc,
    parse_alpha_vantage_quote,
    parse_alpha_vantage_ohlc,
)


def test_parse_alpha_vantage_daily_ohlc_returns_price_frame() -> None:
    payload = {
        "Meta Data": {"2. Symbol": "AAA"},
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "100.0",
                "2. high": "103.0",
                "3. low": "99.0",
                "4. close": "101.5",
                "5. adjusted close": "100.5",
                "6. volume": "1000000",
            },
            "2024-01-01": {
                "1. open": "99.0",
                "2. high": "102.0",
                "3. low": "98.0",
                "4. close": "100.0",
                "5. adjusted close": "99.0",
                "6. volume": "900000",
            },
        },
    }

    frame = parse_alpha_vantage_ohlc("AAA", payload, "1d")

    assert list(frame.columns) == ["High", "Low", "Close", "Adjusted Close", "Volume"]
    assert frame.iloc[-1]["Close"] == 101.5
    assert frame.iloc[-1]["Adjusted Close"] == 100.5
    assert str(frame.index[0].date()) == "2024-01-01"


def test_parse_alpha_vantage_intraday_ohlc_returns_price_frame() -> None:
    payload = {
        "Meta Data": {"2. Symbol": "AAA"},
        "Time Series (15min)": {
            "2024-01-02 09:45:00": {
                "1. open": "100.0",
                "2. high": "103.0",
                "3. low": "99.0",
                "4. close": "101.5",
                "5. volume": "1000000",
            },
            "2024-01-02 09:30:00": {
                "1. open": "99.0",
                "2. high": "102.0",
                "3. low": "98.0",
                "4. close": "100.0",
                "5. volume": "900000",
            },
        },
    }

    frame = parse_alpha_vantage_ohlc("AAA", payload, "15m")

    assert list(frame.columns) == ["High", "Low", "Close", "Volume"]
    assert frame.iloc[-1]["Close"] == 101.5
    assert str(frame.index[-1]) == "2024-01-02 09:45:00"


def test_alpha_vantage_params_uses_intraday_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    params = alpha_vantage_params("AAPL", MarketDataRequest(tickers=["AAPL"], start="2026-04-27", interval="15m"))

    assert params["function"] == "TIME_SERIES_INTRADAY"
    assert params["interval"] == "15min"
    assert params["adjusted"] == "true"
    assert params["extended_hours"] == "false"
    assert params["apikey"] == "test-key"


def test_alpha_vantage_params_uses_free_daily_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    params = alpha_vantage_params("AAPL", MarketDataRequest(tickers=["AAPL"], start="2026-04-27"))

    assert params["function"] == "TIME_SERIES_DAILY"
    assert params["outputsize"] == "full"
    assert params["apikey"] == "test-key"


def test_parse_alpha_vantage_quote_returns_latest_price() -> None:
    payload = {
        "Global Quote": {
            "01. symbol": "AAPL",
            "05. price": "210.1500",
            "07. latest trading day": "2026-04-27",
            "08. previous close": "209.0000",
            "09. change": "1.1500",
            "10. change percent": "0.5502%",
        }
    }

    quote = parse_alpha_vantage_quote("AAPL", payload)

    assert quote.ticker == "AAPL"
    assert quote.price == 210.15
    assert quote.latest_trading_day == "2026-04-27"
    assert quote.previous_close == 209.0
    assert quote.change_percent == "0.5502%"


def test_fetch_ohlc_uses_alpha_vantage_request(monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Time Series (15min)": {
                    "2026-04-27 09:45:00": {
                        "1. open": "100.0",
                        "2. high": "102.0",
                        "3. low": "99.0",
                        "4. close": "101.0",
                        "5. volume": "1000",
                    }
                }
            }

    def fake_get(url, params, timeout):
        captured.update({"url": url, **params})
        return FakeResponse()

    monkeypatch.setattr("trading_app.data.requests.get", fake_get)

    result = fetch_ohlc(MarketDataRequest(tickers=["AAPL"], start="2026-04-27", end="2026-04-28", interval="15m"))

    assert "AAPL" in result
    assert captured["function"] == "TIME_SERIES_INTRADAY"
    assert captured["interval"] == "15min"
    assert captured["apikey"] == "test-key"
