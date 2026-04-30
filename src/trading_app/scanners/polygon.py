from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


POLYGON_BASE_URL = "https://api.polygon.io"


@dataclass(frozen=True)
class PolygonSnapshotScannerConfig:
    min_gap_pct: float = 5.0
    min_volume: int = 100_000
    min_price: float = 2.0
    max_price: float = 20.0
    top_n: int = 50
    include_otc: bool = False


class PolygonClient:
    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY is required for Polygon snapshot scanning")
        self.api_key = api_key
        self.session = session or requests.Session()

    def full_market_snapshot(self, include_otc: bool = False) -> dict[str, Any]:
        response = self.session.get(
            f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers",
            params={
                "include_otc": str(include_otc).lower(),
                "apiKey": self.api_key,
            },
            timeout=60,
        )
        if response.status_code == 403:
            raise PermissionError(
                "Polygon rejected the full-market snapshot request with 403 Forbidden. "
                "Your API key likely does not include access to "
                "/v2/snapshot/locale/us/markets/stocks/tickers. "
                "Use a Polygon plan with stock snapshot access, or switch the scanner provider to Finnhub."
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in {None, "OK", "DELAYED"}:
            raise ValueError(f"Polygon snapshot returned status: {payload.get('status')}")
        return payload


def polygon_scanner_columns() -> list[str]:
    return [
        "ticker",
        "current_price",
        "gap_percent",
        "volume",
        "previous_close",
        "day_open",
        "day_high",
        "day_low",
        "day_close",
        "minute_volume",
        "updated",
        "error",
    ]


def scan_polygon_snapshot(
    api_key: str,
    config: PolygonSnapshotScannerConfig | None = None,
    client: PolygonClient | None = None,
) -> pd.DataFrame:
    scan_config = config or PolygonSnapshotScannerConfig()
    polygon = client or PolygonClient(api_key)
    payload = polygon.full_market_snapshot(include_otc=scan_config.include_otc)
    rows = []
    errors = []
    for snapshot in payload.get("tickers") or []:
        row = parse_snapshot_row(snapshot)
        if row.get("error"):
            errors.append(row)
            continue
        if passes_filters(row, scan_config):
            rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (
            row.get("gap_percent") or float("-inf"),
            row.get("volume") or float("-inf"),
        ),
        reverse=True,
    )[: scan_config.top_n]
    return pd.DataFrame(rows + errors, columns=polygon_scanner_columns())


def parse_snapshot_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    ticker = str(snapshot.get("ticker", "")).strip().upper()
    try:
        day = snapshot.get("day") or {}
        minute = snapshot.get("min") or {}
        previous_day = snapshot.get("prevDay") or {}
        last_trade = snapshot.get("lastTrade") or {}
        current_price = first_number(
            last_trade.get("p"),
            minute.get("c"),
            day.get("c"),
            previous_day.get("c"),
        )
        previous_close = first_number(previous_day.get("c"))
        gap_percent = snapshot.get("todaysChangePerc")
        if gap_percent is None:
            gap_percent = percent_change(current_price, previous_close)
        volume = first_number(day.get("v"), 0)
        return {
            "ticker": ticker,
            "current_price": current_price,
            "gap_percent": float(gap_percent),
            "volume": int(volume),
            "previous_close": previous_close,
            "day_open": optional_number(day.get("o")),
            "day_high": optional_number(day.get("h")),
            "day_low": optional_number(day.get("l")),
            "day_close": optional_number(day.get("c")),
            "minute_volume": optional_int(minute.get("v")),
            "updated": snapshot.get("updated"),
            "error": "",
        }
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def passes_filters(row: dict[str, Any], config: PolygonSnapshotScannerConfig) -> bool:
    return (
        row["gap_percent"] > config.min_gap_pct
        and row["volume"] > config.min_volume
        and config.min_price <= row["current_price"] <= config.max_price
    )


def first_number(*values) -> float:
    for value in values:
        if value is None:
            continue
        parsed = float(value)
        if pd.notna(parsed) and parsed > 0:
            return parsed
    raise ValueError("No positive numeric value found")


def optional_number(value) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_int(value) -> int | None:
    if value is None:
        return None
    return int(float(value))


def percent_change(current: float, reference: float) -> float:
    if reference == 0:
        raise ValueError("Cannot calculate gap from zero previous close")
    return (current - reference) / reference * 100.0
