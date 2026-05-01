from __future__ import annotations

from trading_app.scanners.polygon import (
    PolygonClient,
    PolygonSnapshotScannerConfig,
    parse_snapshot_row,
    scan_polygon_snapshot,
)


class FakePolygonClient:
    def __init__(self):
        self.include_otc = None

    def full_market_snapshot(self, include_otc=False):
        self.include_otc = include_otc
        return {
            "status": "OK",
            "tickers": [
                snapshot("AAA", price=10.0, previous_close=9.0, gap=11.1, volume=500_000),
                snapshot("BBB", price=8.0, previous_close=7.5, gap=6.6, volume=1_000_000),
                snapshot("LOWVOL", price=9.0, previous_close=8.0, gap=12.5, volume=1_000),
                snapshot("PRICEY", price=25.0, previous_close=20.0, gap=25.0, volume=500_000),
                snapshot("FLAT", price=10.1, previous_close=10.0, gap=1.0, volume=500_000),
            ],
        }


def snapshot(ticker, price, previous_close, gap, volume):
    return {
        "ticker": ticker,
        "todaysChangePerc": gap,
        "updated": 1,
        "lastTrade": {"p": price},
        "day": {
            "o": previous_close,
            "h": price + 0.5,
            "l": price - 0.5,
            "c": price,
            "v": volume,
        },
        "min": {"c": price, "v": 1000},
        "prevDay": {"c": previous_close},
    }


def test_polygon_snapshot_scanner_filters_and_sorts_top_movers() -> None:
    client = FakePolygonClient()

    result = scan_polygon_snapshot(
        "key",
        config=PolygonSnapshotScannerConfig(
            min_gap_pct=5.0,
            min_volume=100_000,
            min_price=2.0,
            max_price=20.0,
            top_n=50,
        ),
        client=client,
    )

    assert list(result["ticker"]) == ["AAA", "BBB"]
    assert result.iloc[0]["gap_percent"] == 11.1
    assert result.iloc[0]["volume"] == 500_000
    assert client.include_otc is False


def test_parse_snapshot_row_calculates_gap_when_missing() -> None:
    row = parse_snapshot_row(
        {
            "ticker": "AAA",
            "lastTrade": {"p": 11.0},
            "day": {"c": 11.0, "v": 200_000},
            "prevDay": {"c": 10.0},
        }
    )

    assert row["gap_percent"] == 10.0
    assert row["current_price"] == 11.0
    assert row["previous_close"] == 10.0


def test_polygon_client_sends_api_key_and_include_otc() -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "OK", "tickers": []}

    class FakeSession:
        def __init__(self):
            self.params = None

        def get(self, url, params, timeout):
            self.params = params
            return FakeResponse()

    session = FakeSession()
    client = PolygonClient("polygon-key", session=session)

    client.full_market_snapshot(include_otc=True)

    assert session.params["apiKey"] == "polygon-key"
    assert session.params["include_otc"] == "true"


def test_polygon_client_translates_snapshot_forbidden() -> None:
    class FakeResponse:
        status_code = 403

        def raise_for_status(self):
            raise AssertionError("raise_for_status should not be called for translated 403")

    class FakeSession:
        def get(self, url, params, timeout):
            return FakeResponse()

    client = PolygonClient("polygon-key", session=FakeSession())

    try:
        client.full_market_snapshot()
    except PermissionError as exc:
        assert "does not include access" in str(exc)
        assert "full-market snapshot" in str(exc)
    else:
        raise AssertionError("Expected PermissionError")
