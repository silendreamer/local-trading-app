from __future__ import annotations

from trading_app.scanners.fmp import FmpClient, FmpGainersScannerConfig, parse_gainer_row, scan_fmp_gainers


class FakeFmpClient:
    def biggest_gainers(self):
        return [
            gainer("AAA", 10.0, "12.5%"),
            gainer("BBB", 8.0, "7.5%"),
            gainer("LOWVOL", 9.0, "15%"),
            gainer("PRICEY", 25.0, "20%"),
            gainer("FLAT", 10.0, "1%"),
        ]

    def batch_quote(self, symbols):
        return [
            {"symbol": "AAA", "volume": 500_000},
            {"symbol": "BBB", "volume": 1_000_000},
            {"symbol": "LOWVOL", "volume": 1_000},
            {"symbol": "PRICEY", "volume": 500_000},
            {"symbol": "FLAT", "volume": 500_000},
        ]


def gainer(symbol, price, change_percent):
    return {
        "symbol": symbol,
        "name": f"{symbol} Corp",
        "price": price,
        "changesPercentage": change_percent,
        "change": 1.0,
    }


def test_fmp_gainers_scanner_filters_and_sorts() -> None:
    result = scan_fmp_gainers(
        "key",
        config=FmpGainersScannerConfig(
            min_gap_pct=5.0,
            min_volume=100_000,
            min_price=2.0,
            max_price=20.0,
            top_n=50,
        ),
        client=FakeFmpClient(),
    )

    assert list(result["ticker"]) == ["AAA", "BBB"]
    assert result.iloc[0]["gap_percent"] == 12.5
    assert result.iloc[0]["volume"] == 500_000


def test_parse_gainer_row_handles_numeric_strings() -> None:
    row = parse_gainer_row(
        {
            "symbol": "AAA",
            "price": "$10.50",
            "changesPercentage": "6.25%",
            "volume": "1,250,000",
        }
    )

    assert row["ticker"] == "AAA"
    assert row["current_price"] == 10.5
    assert row["gap_percent"] == 6.25
    assert row["volume"] == 1_250_000


def test_fmp_client_sends_api_key() -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    class FakeSession:
        def __init__(self):
            self.params = None

        def get(self, url, params, timeout):
            self.params = params
            return FakeResponse()

    session = FakeSession()
    client = FmpClient("fmp-key", session=session)

    client.biggest_gainers()

    assert session.params["apikey"] == "fmp-key"


def test_fmp_client_batch_quote_sends_symbols() -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, params, timeout):
            self.calls.append((url, params))
            return FakeResponse()

    session = FakeSession()
    client = FmpClient("fmp-key", session=session)

    client.batch_quote(["AAA", "BBB"])

    assert session.calls[0][1]["apikey"] == "fmp-key"
    assert session.calls[0][1]["symbols"] == "AAA,BBB"
