from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


FMP_BASE_URL = "https://financialmodelingprep.com"


@dataclass(frozen=True)
class FmpGainersScannerConfig:
    min_gap_pct: float = 5.0
    min_volume: int = 100_000
    min_price: float = 2.0
    max_price: float = 20.0
    top_n: int = 50


class FmpClient:
    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        if not api_key:
            raise ValueError("FMP_API_KEY is required for Financial Modeling Prep scanning")
        self.api_key = api_key
        self.session = session or requests.Session()

    def biggest_gainers(self) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{FMP_BASE_URL}/stable/biggest-gainers",
            params={"apikey": self.api_key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise ValueError(str(payload["Error Message"]))
        if not isinstance(payload, list):
            raise ValueError("FMP biggest-gainers response was not a list")
        return payload

    def batch_quote(self, symbols: list[str]) -> list[dict[str, Any]]:
        if not symbols:
            return []
        response = self.session.get(
            f"{FMP_BASE_URL}/stable/batch-quote",
            params={
                "symbols": ",".join(symbols),
                "apikey": self.api_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise ValueError(str(payload["Error Message"]))
        if not isinstance(payload, list):
            raise ValueError("FMP batch-quote response was not a list")
        return payload


def fmp_scanner_columns() -> list[str]:
    return [
        "ticker",
        "name",
        "current_price",
        "gap_percent",
        "volume",
        "change",
        "error",
    ]


def scan_fmp_gainers(
    api_key: str,
    config: FmpGainersScannerConfig | None = None,
    client: FmpClient | None = None,
) -> pd.DataFrame:
    scan_config = config or FmpGainersScannerConfig()
    fmp = client or FmpClient(api_key)
    gainers = fmp.biggest_gainers()
    symbols = [
        str(first_present(item, "symbol", "ticker") or "").strip().upper()
        for item in gainers
    ]
    quote_by_symbol = {
        str(item.get("symbol", "")).strip().upper(): item
        for item in fmp.batch_quote([symbol for symbol in symbols if symbol])
    }
    rows = []
    errors = []
    for item in gainers:
        symbol = str(first_present(item, "symbol", "ticker") or "").strip().upper()
        row = parse_gainer_row({**item, **quote_by_symbol.get(symbol, {})})
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
    return pd.DataFrame(rows + errors, columns=fmp_scanner_columns())


def parse_gainer_row(item: dict[str, Any]) -> dict[str, Any]:
    ticker = str(first_present(item, "symbol", "ticker") or "").strip().upper()
    try:
        price = parse_float(first_present(item, "price", "currentPrice", "lastPrice"))
        gap_percent = parse_percent(first_present(item, "changesPercentage", "changePercentage", "percentChange"))
        volume = parse_int(first_present(item, "volume", "vol", default=0))
        return {
            "ticker": ticker,
            "name": first_present(item, "name", "companyName", default=""),
            "current_price": price,
            "gap_percent": gap_percent,
            "volume": volume,
            "change": parse_optional_float(first_present(item, "change", "changeAmount")),
            "error": "",
        }
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def passes_filters(row: dict[str, Any], config: FmpGainersScannerConfig) -> bool:
    return (
        row["gap_percent"] > config.min_gap_pct
        and row["volume"] > config.min_volume
        and config.min_price <= row["current_price"] <= config.max_price
    )


def first_present(item: dict[str, Any], *keys: str, default=None):
    for key in keys:
        value = item.get(key)
        if value not in {None, ""}:
            return value
    return default


def parse_float(value) -> float:
    return float(str(value).replace(",", "").replace("$", "").strip())


def parse_optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    return parse_float(value)


def parse_int(value) -> int:
    return int(float(str(value).replace(",", "").strip()))


def parse_percent(value) -> float:
    return float(str(value).replace("%", "").replace(",", "").strip())
