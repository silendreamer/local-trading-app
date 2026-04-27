from __future__ import annotations

from trading_app.data import parse_yahoo_chart_close, parse_yahoo_chart_ohlc


def test_parse_yahoo_chart_close_returns_price_series() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704067200, 1704153600],
                    "indicators": {
                        "quote": [
                            {
                                "close": [100.0, 101.5],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    series = parse_yahoo_chart_close("AAA", payload)

    assert series.name == "AAA"
    assert series.iloc[-1] == 101.5
    assert str(series.index[0].date()) == "2024-01-01"


def test_parse_yahoo_chart_ohlc_returns_price_frame() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704067200, 1704153600],
                    "indicators": {
                        "quote": [
                            {
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [100.0, 101.5],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }

    frame = parse_yahoo_chart_ohlc("AAA", payload)

    assert list(frame.columns) == ["High", "Low", "Close"]
    assert frame.iloc[-1]["Close"] == 101.5
    assert str(frame.index[0].date()) == "2024-01-01"
