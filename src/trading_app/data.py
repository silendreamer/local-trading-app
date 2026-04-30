from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os

import pandas as pd
import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
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
    """Fetch adjusted close prices from Alpha Vantage."""
    frames = [
        fetch_ticker_prices_from_alpha_vantage(ticker, request)
        for ticker in validate_tickers(request.tickers)
    ]
    prices = pd.concat(frames, axis=1).reindex(columns=request.tickers)
    prices = filter_date_range(prices, request.start, request.end)
    if prices.empty or prices.dropna(how="all").empty:
        raise ValueError("No market data returned")
    return prices.dropna(how="all").sort_index()


def fetch_ohlc(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
    """Fetch OHLC candles from Alpha Vantage keyed by ticker."""
    result = {
        ticker: filter_date_range(
            fetch_ticker_ohlc_from_alpha_vantage(ticker, request),
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


def alpha_vantage_api_key() -> str:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is required for market data")
    return api_key


def fetch_ticker_prices_from_alpha_vantage(ticker: str, request: MarketDataRequest) -> pd.Series:
    frame = fetch_ticker_ohlc_from_alpha_vantage(ticker, request)
    if request.interval == "1d" and "Adjusted Close" in frame:
        return frame["Adjusted Close"].rename(ticker)
    return frame["Close"].rename(ticker)


def fetch_ticker_ohlc_from_alpha_vantage(ticker: str, request: MarketDataRequest) -> pd.DataFrame:
    payload = request_alpha_vantage(ticker, request)
    return parse_alpha_vantage_ohlc(ticker, payload, request.interval)


def fetch_latest_quote(ticker: str) -> LatestQuote:
    normalized = validate_tickers([ticker])[0]
    response = requests.get(
        ALPHA_VANTAGE_URL,
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": normalized,
            "apikey": alpha_vantage_api_key(),
            "datatype": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    return parse_alpha_vantage_quote(normalized, response.json())


def request_alpha_vantage(ticker: str, request: MarketDataRequest) -> dict:
    params = alpha_vantage_params(ticker, request)
    response = requests.get(
        ALPHA_VANTAGE_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def alpha_vantage_params(ticker: str, request: MarketDataRequest) -> dict[str, str]:
    interval = normalize_interval(request.interval)
    params = {
        "symbol": ticker,
        "apikey": alpha_vantage_api_key(),
        "datatype": "json",
    }
    if interval == "1d":
        params.update(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "outputsize": "full",
            }
        )
    elif interval in INTRADAY_INTERVALS:
        params.update(
            {
                "function": "TIME_SERIES_INTRADAY",
                "interval": interval,
                "outputsize": "compact",
                "adjusted": "true",
                "extended_hours": "false",
            }
        )
    else:
        supported = ", ".join(["1d", *sorted(INTRADAY_INTERVALS)])
        raise ValueError(f"Unsupported Alpha Vantage interval '{request.interval}'. Supported intervals: {supported}")
    return params


def normalize_interval(interval: str) -> str:
    return INTERVAL_ALIASES.get(interval, interval)


def parse_alpha_vantage_ohlc(ticker: str, payload: dict, interval: str) -> pd.DataFrame:
    raise_for_alpha_vantage_error(ticker, payload)
    series_key = time_series_key(payload)
    rows = payload.get(series_key) or {}
    if not rows:
        raise ValueError(f"{ticker}: no Alpha Vantage time series returned")

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.to_datetime(frame.index)
    rename_map = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adjusted Close",
        "5. volume": "Volume",
        "6. volume": "Volume",
    }
    frame = frame.rename(columns=rename_map)
    required = ["High", "Low", "Close"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: Alpha Vantage data is missing columns: {', '.join(sorted(missing))}")

    keep_columns = required.copy()
    if normalize_interval(interval) == "1d" and "Adjusted Close" in frame.columns:
        keep_columns.append("Adjusted Close")
    if "Volume" in frame.columns:
        keep_columns.append("Volume")
    frame = frame.loc[:, keep_columns].apply(pd.to_numeric, errors="coerce").dropna(subset=required)
    if frame.empty:
        raise ValueError(f"{ticker}: no OHLC prices returned")
    return frame.sort_index()


def parse_alpha_vantage_quote(ticker: str, payload: dict) -> LatestQuote:
    raise_for_alpha_vantage_error(ticker, payload)
    quote = payload.get("Global Quote") or {}
    if not quote:
        raise ValueError(f"{ticker}: no Alpha Vantage quote returned")

    price = quote.get("05. price")
    latest_trading_day = quote.get("07. latest trading day", "")
    if not price:
        raise ValueError(f"{ticker}: Alpha Vantage quote did not include a price")

    return LatestQuote(
        ticker=quote.get("01. symbol", ticker),
        price=float(price),
        latest_trading_day=latest_trading_day,
        previous_close=parse_optional_float(quote.get("08. previous close")),
        change=parse_optional_float(quote.get("09. change")),
        change_percent=quote.get("10. change percent", ""),
    )


def parse_optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def raise_for_alpha_vantage_error(ticker: str, payload: dict) -> None:
    for key in ("Error Message", "Information", "Note"):
        message = payload.get(key)
        if message:
            raise ValueError(f"{ticker}: Alpha Vantage returned {key}: {message}")


def time_series_key(payload: dict) -> str:
    for key in payload:
        if key.startswith("Time Series"):
            return key
    raise ValueError("Alpha Vantage response did not include a time series")


def filter_date_range(frame: pd.DataFrame | pd.Series, start: date | str, end: date | str | None):
    filtered = frame.copy()
    index = pd.to_datetime(filtered.index)
    start_ts = pd.Timestamp(start)
    filtered = filtered.loc[index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
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
