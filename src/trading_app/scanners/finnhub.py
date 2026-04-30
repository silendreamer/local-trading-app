from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
import json
from pathlib import Path
import time as clock
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from trading_app.config import PROJECT_ROOT


FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
PREMARKET_START = time(4, 0)
REGULAR_MARKET_START = time(9, 30)


@dataclass(frozen=True)
class FinnhubScannerConfig:
    min_gap_pct: float = 5.0
    min_premarket_volume: int = 100_000
    min_price: float = 2.0
    max_price: float = 20.0
    top_n: int = 20
    quote_batch_size: int = 100
    candle_batch_size: int = 25
    requests_per_minute: int = 60
    candle_resolution: str = "1"


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.min_interval = 0.0 if requests_per_minute <= 0 else 60.0 / requests_per_minute
        self.last_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = clock.monotonic()
        wait_seconds = self.min_interval - (now - self.last_request_at)
        if wait_seconds > 0:
            clock.sleep(wait_seconds)
        self.last_request_at = clock.monotonic()


class FinnhubClient:
    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        requests_per_minute: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is required for the scanner")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.rate_limiter = RateLimiter(requests_per_minute)

    def stock_symbols(self, exchange: str = "US") -> list[dict[str, Any]]:
        return self._get("/stock/symbol", {"exchange": exchange})

    def quote(self, symbol: str) -> dict[str, Any]:
        return self._get("/quote", {"symbol": symbol})

    def stock_candles(self, symbol: str, resolution: str, start_ts: int, end_ts: int) -> dict[str, Any]:
        return self._get(
            "/stock/candle",
            {
                "symbol": symbol,
                "resolution": resolution,
                "from": start_ts,
                "to": end_ts,
            },
        )

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        self.rate_limiter.wait()
        response = self.session.get(
            f"{FINNHUB_BASE_URL}{path}",
            params={**params, "token": self.api_key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(str(payload["error"]))
        return payload


def scanner_columns() -> list[str]:
    return [
        "ticker",
        "previous_day_high",
        "previous_day_close",
        "previous_day_volume",
        "pre_market_high",
        "pre_market_volume",
        "current_price",
        "percent_gap_from_previous_close",
        "percent_move_from_previous_day_high",
        "relative_volume",
        "above_pre_market_high",
        "latest_price_timestamp",
        "continuation_signal",
        "continuation_reason",
        "error",
    ]


def scan_finnhub_momentum(
    api_key: str,
    config: FinnhubScannerConfig | None = None,
    now: datetime | None = None,
    cache_dir: Path | None = None,
    client: FinnhubClient | None = None,
) -> pd.DataFrame:
    scan_config = config or FinnhubScannerConfig()
    current_time = normalize_market_time(now)
    finnhub = client or FinnhubClient(api_key, requests_per_minute=scan_config.requests_per_minute)
    symbols = load_or_fetch_us_symbols(finnhub, cache_dir=cache_dir, now=current_time)

    quote_candidates = []
    quote_errors = []
    for chunk in chunked(symbols, scan_config.quote_batch_size):
        for symbol in chunk:
            try:
                quote_row = quote_prefilter_row(symbol, finnhub.quote(symbol), current_time)
                if quote_row["percent_gap_from_previous_close"] >= scan_config.min_gap_pct:
                    quote_candidates.append(quote_row)
            except Exception as exc:
                quote_errors.append({"ticker": symbol, "error": str(exc)})

    rows = []
    candle_errors = []
    for chunk in chunked([row["ticker"] for row in quote_candidates], scan_config.candle_batch_size):
        for symbol in chunk:
            quote_row = next(row for row in quote_candidates if row["ticker"] == symbol)
            try:
                rows.append(enrich_with_candles(symbol, quote_row, finnhub, scan_config, current_time))
            except Exception as exc:
                candle_errors.append({"ticker": symbol, "error": str(exc)})

    rows = [row for row in rows if passes_final_filters(row, scan_config)]
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("percent_gap_from_previous_close") or float("-inf"),
            row.get("relative_volume") or float("-inf"),
        ),
        reverse=True,
    )[: scan_config.top_n]
    return pd.DataFrame(rows + candle_errors + quote_errors, columns=scanner_columns())


def load_or_fetch_us_symbols(
    client: FinnhubClient,
    cache_dir: Path | None = None,
    now: datetime | None = None,
) -> list[str]:
    current_time = normalize_market_time(now)
    target_dir = cache_dir or PROJECT_ROOT / "data" / "finnhub-cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_path = target_dir / f"us-symbols-{current_time.date().isoformat()}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return normalize_symbols(payload.get("symbols", []))

    symbols = normalize_symbols(client.stock_symbols("US"))
    cache_path.write_text(json.dumps({"date": current_time.date().isoformat(), "symbols": symbols}), encoding="utf-8")
    return symbols


def normalize_symbols(raw_symbols: list[Any]) -> list[str]:
    symbols = []
    seen = set()
    for item in raw_symbols:
        if isinstance(item, dict):
            symbol = str(item.get("symbol", "")).strip().upper()
            currency = str(item.get("currency", "")).upper()
            security_type = str(item.get("type", "")).lower()
            if currency and currency != "USD":
                continue
            if security_type and not any(token in security_type for token in ["common", "adr", "reit", "etp", "fund"]):
                continue
        else:
            symbol = str(item).strip().upper()
        if not is_scan_symbol(symbol) or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def is_scan_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    blocked = [" ", "/", "^", "="]
    return not any(token in symbol for token in blocked)


def quote_prefilter_row(symbol: str, quote: dict[str, Any], now: datetime) -> dict[str, Any]:
    current_price = parse_positive_float(quote.get("c"), "current price")
    previous_close = parse_positive_float(quote.get("pc"), "previous close")
    timestamp = int(quote.get("t") or 0)
    return {
        "ticker": symbol,
        "current_price": current_price,
        "previous_day_close": previous_close,
        "percent_gap_from_previous_close": percent_change(current_price, previous_close) or 0.0,
        "latest_price_timestamp": timestamp_to_market_text(timestamp) if timestamp else str(now),
    }


def enrich_with_candles(
    symbol: str,
    quote_row: dict[str, Any],
    client: FinnhubClient,
    config: FinnhubScannerConfig,
    now: datetime,
) -> dict[str, Any]:
    market_open = datetime.combine(now.date(), PREMARKET_START, MARKET_TIMEZONE)
    daily_start = int((now - pd.Timedelta(days=15)).timestamp())
    daily_end = int(now.timestamp())
    intraday = candles_to_frame(
        client.stock_candles(symbol, config.candle_resolution, int(market_open.timestamp()), int(now.timestamp()))
    )
    daily = candles_to_frame(client.stock_candles(symbol, "D", daily_start, daily_end))
    previous_day = previous_trading_day_row(daily, now)
    premarket = premarket_candles(intraday)
    if premarket.empty:
        raise ValueError("No premarket candles for today")

    previous_high = float(previous_day["High"])
    previous_volume = int(previous_day["Volume"])
    premarket_high = float(premarket["High"].max())
    premarket_volume = int(premarket["Volume"].sum())
    current_price = float(quote_row["current_price"])
    signal, reason = continuation_signal(current_price, previous_high, premarket_high)
    row = {
        **quote_row,
        "previous_day_high": previous_high,
        "previous_day_volume": previous_volume,
        "pre_market_high": premarket_high,
        "pre_market_volume": premarket_volume,
        "percent_move_from_previous_day_high": percent_change(current_price, previous_high),
        "relative_volume": premarket_volume / previous_volume if previous_volume else None,
        "above_pre_market_high": current_price > premarket_high,
        "continuation_signal": signal,
        "continuation_reason": reason,
        "error": "",
    }
    return row


def candles_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    status = payload.get("s")
    if status and status != "ok":
        raise ValueError(f"Finnhub candle response status: {status}")
    timestamps = payload.get("t") or []
    if not timestamps:
        raise ValueError("No candle timestamps returned")
    frame = pd.DataFrame(
        {
            "Open": payload.get("o") or [],
            "High": payload.get("h") or [],
            "Low": payload.get("l") or [],
            "Close": payload.get("c") or [],
            "Volume": payload.get("v") or [],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(MARKET_TIMEZONE),
    )
    return frame.apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Close", "Volume"]).sort_index()


def previous_trading_day_row(daily: pd.DataFrame, now: datetime) -> pd.Series:
    if daily.empty:
        raise ValueError("No daily candles returned")
    previous_days = daily[daily.index.date < now.date()]
    if previous_days.empty:
        raise ValueError("No previous trading day found")
    return previous_days.sort_index().iloc[-1]


def premarket_candles(intraday: pd.DataFrame) -> pd.DataFrame:
    today = intraday.copy()
    clock_values = pd.Series(today.index.time, index=today.index)
    return today[(clock_values >= PREMARKET_START) & (clock_values < REGULAR_MARKET_START)]


def passes_final_filters(row: dict[str, Any], config: FinnhubScannerConfig) -> bool:
    price = row.get("current_price")
    volume = row.get("pre_market_volume")
    return (
        price is not None
        and volume is not None
        and config.min_price <= price <= config.max_price
        and volume >= config.min_premarket_volume
    )


def continuation_signal(
    current_price: float,
    previous_day_high: float,
    premarket_high: float,
) -> tuple[str, str]:
    if current_price > premarket_high and current_price > previous_day_high:
        return "BREAKOUT_CONTINUATION", "Current price is above both premarket high and previous day high"
    if current_price > premarket_high:
        return "PREMARKET_HIGH_BREAK", "Current price is above premarket high"
    if current_price > previous_day_high:
        return "PREVIOUS_HIGH_BREAK", "Current price is above previous day high"
    return "WATCH", "Current price has not cleared premarket or previous day high"


def parse_positive_float(value: Any, label: str) -> float:
    parsed = float(value or 0.0)
    if parsed <= 0:
        raise ValueError(f"Missing valid {label}")
    return parsed


def percent_change(current: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return (current - reference) / reference * 100.0


def timestamp_to_market_text(timestamp: int) -> str:
    return str(datetime.fromtimestamp(timestamp, tz=MARKET_TIMEZONE))


def normalize_market_time(now: datetime | None = None) -> datetime:
    current_time = now or datetime.now(MARKET_TIMEZONE)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=MARKET_TIMEZONE)
    return current_time.astimezone(MARKET_TIMEZONE)


def chunked(values: list[str], size: int):
    safe_size = max(int(size), 1)
    for index in range(0, len(values), safe_size):
        yield values[index : index + safe_size]
