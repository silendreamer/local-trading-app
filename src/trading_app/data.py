from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os

import pandas as pd
import requests


POLYGON_BASE_URL = "https://api.polygon.io"
INTRADAY_INTERVALS = {"1min", "5min", "15min", "30min", "60min"}
INTERVAL_ALIASES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1h": "60min",
}


@dataclass(frozen=True)
class MarketDataRequest:
    tickers: list[str]
    start: date | str
    end: date | str | None = None
    interval: str = "1d"


@dataclass(frozen=True)
class LatestQuote:
    ticker: str
    price: float
    latest_trading_day: str
    previous_close: float | None = None
    change: float | None = None
    change_percent: str = ""


def fetch_prices(request: MarketDataRequest) -> pd.DataFrame:
    """Fetch close prices from Polygon."""
    frames = [
        fetch_ticker_prices_from_polygon(ticker, request)
        for ticker in validate_tickers(request.tickers)
    ]
    prices = pd.concat(frames, axis=1).reindex(columns=request.tickers)
    prices = filter_date_range(prices, request.start, request.end)
    if prices.empty or prices.dropna(how="all").empty:
        raise ValueError("No market data returned")
    return prices.dropna(how="all").sort_index()


def fetch_ohlc(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
    """Fetch Polygon OHLC candles keyed by ticker."""
    result = {
        ticker: filter_date_range(
            fetch_ticker_ohlc_from_polygon(ticker, request),
            request.start,
            request.end,
        )
        for ticker in validate_tickers(request.tickers)
    }
    result = {ticker: frame for ticker, frame in result.items() if not frame.empty}
    if not result:
        raise ValueError("No market data returned")
    return result


def validate_tickers(tickers: list[str]) -> list[str]:
    normalized = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized:
        raise ValueError("At least one ticker is required")
    return normalized


def polygon_api_key() -> str:
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise ValueError("POLYGON_API_KEY is required for market data")
    return api_key


def fetch_ticker_prices_from_polygon(ticker: str, request: MarketDataRequest) -> pd.Series:
    frame = fetch_ticker_ohlc_from_polygon(ticker, request)
    return frame["Close"].rename(ticker)


def fetch_ticker_ohlc_from_polygon(ticker: str, request: MarketDataRequest) -> pd.DataFrame:
    payload = request_polygon_aggregates(ticker, request)
    return parse_polygon_ohlc(ticker, payload, request.interval)


def fetch_latest_quote(ticker: str) -> LatestQuote:
    normalized = validate_tickers([ticker])[0]
    payload = request_polygon_snapshot(normalized)
    return parse_polygon_snapshot_quote(normalized, payload)


def request_polygon_aggregates(ticker: str, request: MarketDataRequest) -> dict:
    multiplier, timespan = polygon_interval_parts(request.interval)
    start = str(pd.Timestamp(request.start).date())
    end_value = request.end or date.today()
    end = str(pd.Timestamp(end_value).date())
    response = requests.get(
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}",
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": polygon_api_key(),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def request_polygon_snapshot(ticker: str) -> dict:
    response = requests.get(
        f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
        params={"apiKey": polygon_api_key()},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def polygon_interval_parts(interval: str) -> tuple[int, str]:
    normalized = normalize_interval(interval)
    if normalized == "1d":
        return 1, "day"
    if normalized in INTRADAY_INTERVALS:
        return int(normalized.removesuffix("min")), "minute"
    supported = ", ".join(["1d", *sorted(INTRADAY_INTERVALS)])
    raise ValueError(f"Unsupported Polygon interval '{interval}'. Supported intervals: {supported}")


def normalize_interval(interval: str) -> str:
    return INTERVAL_ALIASES.get(interval, interval)


def parse_polygon_ohlc(ticker: str, payload: dict, interval: str = "1d") -> pd.DataFrame:
    raise_for_polygon_error(ticker, payload)
    rows = payload.get("results") or []
    if not rows:
        raise ValueError(f"{ticker}: no Polygon aggregate bars returned")

    frame = pd.DataFrame.from_records(rows)
    required = {"h", "l", "c"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: Polygon data is missing columns: {', '.join(sorted(missing))}")

    if normalize_interval(interval) == "1d":
        frame.index = pd.to_datetime(pd.to_datetime(frame["t"], unit="ms", utc=True).dt.date)
    else:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame["t"], unit="ms", utc=True)).tz_convert("America/New_York")
    rename_map = {
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "vw": "VWAP",
        "n": "Transactions",
    }
    frame = frame.rename(columns=rename_map)
    keep_columns = [column for column in ["Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"] if column in frame]
    frame = frame.loc[:, keep_columns].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])
    if frame.empty:
        raise ValueError(f"{ticker}: no OHLC prices returned")
    return frame.sort_index()


def parse_polygon_snapshot_quote(ticker: str, payload: dict) -> LatestQuote:
    raise_for_polygon_error(ticker, payload)
    snapshot = payload.get("ticker") or {}
    if not snapshot:
        raise ValueError(f"{ticker}: no Polygon snapshot returned")

    last_trade = snapshot.get("lastTrade") or {}
    minute = snapshot.get("min") or {}
    day = snapshot.get("day") or {}
    previous_day = snapshot.get("prevDay") or {}
    price = first_number(last_trade.get("p"), minute.get("c"), day.get("c"), previous_day.get("c"))
    previous_close = optional_float(previous_day.get("c"))
    change = optional_float(snapshot.get("todaysChange"))
    change_percent = snapshot.get("todaysChangePerc")
    timestamp = first_value(last_trade.get("t"), minute.get("t"), day.get("t"))
    latest_trading_day = ""
    if timestamp is not None:
        latest_trading_day = str(parse_polygon_timestamp(timestamp).date())

    return LatestQuote(
        ticker=str(snapshot.get("ticker") or ticker).upper(),
        price=price,
        latest_trading_day=latest_trading_day,
        previous_close=previous_close,
        change=change,
        change_percent="" if change_percent is None else f"{float(change_percent):.4f}%",
    )


def raise_for_polygon_error(ticker: str, payload: dict) -> None:
    status = payload.get("status")
    if status in {"ERROR", "NOT_AUTHORIZED"}:
        message = payload.get("error") or payload.get("message") or payload
        raise ValueError(f"{ticker}: Polygon returned {status}: {message}")


def first_number(*values) -> float:
    for value in values:
        parsed = optional_float(value)
        if parsed is not None and pd.notna(parsed) and parsed > 0:
            return parsed
    raise ValueError("No positive numeric value found")


def first_value(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_polygon_timestamp(value) -> pd.Timestamp:
    timestamp = int(value)
    unit = "ns" if timestamp > 10_000_000_000_000_000 else "ms"
    return pd.to_datetime(timestamp, unit=unit, utc=True)


def optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def filter_date_range(frame: pd.DataFrame | pd.Series, start: date | str, end: date | str | None):
    filtered = frame.copy()
    index = pd.to_datetime(filtered.index)
    start_ts = pd.Timestamp(start, tz=index.tz)
    filtered = filtered.loc[index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end, tz=index.tz)
        filtered = filtered.loc[pd.to_datetime(filtered.index) < end_ts]
    return filtered.sort_index()


def latest_prices(prices: pd.DataFrame) -> pd.Series:
    """Return the most recent non-null price for each ticker."""
    if prices.empty:
        raise ValueError("Price data is empty")
    latest = prices.ffill().iloc[-1].dropna()
    if latest.empty:
        raise ValueError("No latest prices available")
    return latest
