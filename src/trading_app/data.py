from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timezone

import pandas as pd
import requests
import yfinance as yf

from trading_app.config import PROJECT_ROOT


@dataclass(frozen=True)
class MarketDataRequest:
    tickers: list[str]
    start: date | str
    end: date | str | None = None
    interval: str = "1d"


def fetch_prices(request: MarketDataRequest) -> pd.DataFrame:
    """Fetch adjusted close prices from yfinance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    if not request.tickers:
        raise ValueError("At least one ticker is required")

    cache_dir = PROJECT_ROOT / "data" / "yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    raw = yf.download(
        tickers=request.tickers,
        start=request.start,
        end=request.end,
        interval=request.interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    try:
        return extract_close_prices(raw, request.tickers)
    except ValueError as exc:
        if "No market data returned" not in str(exc):
            raise
        return fetch_prices_from_yahoo_chart(request)


def fetch_ohlc(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
    """Fetch OHLC data keyed by ticker.

    yfinance is attempted first. If it returns no usable data, the direct Yahoo
    chart endpoint is used as a fallback.
    """
    if not request.tickers:
        raise ValueError("At least one ticker is required")

    cache_dir = PROJECT_ROOT / "data" / "yfinance-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    raw = yf.download(
        tickers=request.tickers,
        start=request.start,
        end=request.end,
        interval=request.interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    try:
        return extract_ohlc(raw, request.tickers)
    except ValueError as exc:
        if "No market data returned" not in str(exc):
            raise
        return fetch_ohlc_from_yahoo_chart(request)


def extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into a close-price DataFrame."""
    if raw.empty:
        raise ValueError("No market data returned")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Market data does not include Close prices")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Market data does not include Close prices")
        ticker = tickers[0] if len(tickers) == 1 else "Close"
        prices = raw[["Close"]].rename(columns={"Close": ticker})

    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index)
    return prices.dropna(how="all").sort_index()


def extract_ohlc(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Normalize yfinance output into per-ticker OHLC DataFrames."""
    if raw.empty:
        raise ValueError("No market data returned")

    required = ["High", "Low", "Close"]
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for column in required:
            if column not in raw.columns.get_level_values(0):
                raise ValueError(f"Market data does not include {column} prices")
        for ticker in tickers:
            ticker_frame = raw.loc[:, pd.IndexSlice[required, ticker]].copy()
            ticker_frame.columns = required
            ticker_frame.index = pd.to_datetime(ticker_frame.index)
            ticker_frame = ticker_frame.dropna(how="any").sort_index()
            if not ticker_frame.empty:
                result[ticker] = ticker_frame
    else:
        missing = set(required) - set(raw.columns)
        if missing:
            raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
        ticker = tickers[0]
        ticker_frame = raw.loc[:, required].copy()
        ticker_frame.index = pd.to_datetime(ticker_frame.index)
        ticker_frame = ticker_frame.dropna(how="any").sort_index()
        if not ticker_frame.empty:
            result[ticker] = ticker_frame

    if not result:
        raise ValueError("No market data returned")
    return result


def fetch_prices_from_yahoo_chart(request: MarketDataRequest) -> pd.DataFrame:
    """Fallback Yahoo chart fetcher used when yfinance returns empty data."""
    if request.interval != "1d":
        raise ValueError("Yahoo chart fallback currently supports only 1d interval")

    frames = [
        fetch_ticker_from_yahoo_chart(ticker, request.start, request.end)
        for ticker in request.tickers
    ]
    prices = pd.concat(frames, axis=1).reindex(columns=request.tickers)
    if prices.empty or prices.dropna(how="all").empty:
        raise ValueError("No market data returned")
    return prices.dropna(how="all").sort_index()


def fetch_ohlc_from_yahoo_chart(request: MarketDataRequest) -> dict[str, pd.DataFrame]:
    """Fallback Yahoo chart OHLC fetcher used when yfinance returns empty data."""
    if request.interval != "1d":
        raise ValueError("Yahoo chart fallback currently supports only 1d interval")

    result = {
        ticker: fetch_ticker_ohlc_from_yahoo_chart(ticker, request.start, request.end)
        for ticker in request.tickers
    }
    result = {ticker: frame for ticker, frame in result.items() if not frame.empty}
    if not result:
        raise ValueError("No market data returned")
    return result


def fetch_ticker_from_yahoo_chart(
    ticker: str,
    start: date | str,
    end: date | str | None,
) -> pd.Series:
    start_ts = _to_unix_timestamp(start, end_of_day=False)
    end_ts = _to_unix_timestamp(end or date.today(), end_of_day=True)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    response = requests.get(
        url,
        params={
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return parse_yahoo_chart_close(ticker, payload)


def fetch_ticker_ohlc_from_yahoo_chart(
    ticker: str,
    start: date | str,
    end: date | str | None,
) -> pd.DataFrame:
    start_ts = _to_unix_timestamp(start, end_of_day=False)
    end_ts = _to_unix_timestamp(end or date.today(), end_of_day=True)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    response = requests.get(
        url,
        params={
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return parse_yahoo_chart_ohlc(ticker, payload)


def parse_yahoo_chart_close(ticker: str, payload: dict) -> pd.Series:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description", "unknown Yahoo chart error")
        raise ValueError(f"{ticker}: {description}")

    results = chart.get("result") or []
    if not results:
        raise ValueError(f"{ticker}: no Yahoo chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    if not timestamps or not closes:
        raise ValueError(f"{ticker}: no close prices returned")

    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
    return pd.Series(closes, index=index, name=ticker, dtype="float64").dropna()


def parse_yahoo_chart_ohlc(ticker: str, payload: dict) -> pd.DataFrame:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description", "unknown Yahoo chart error")
        raise ValueError(f"{ticker}: {description}")

    results = chart.get("result") or []
    if not results:
        raise ValueError(f"{ticker}: no Yahoo chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps:
        raise ValueError(f"{ticker}: no timestamps returned")

    frame = pd.DataFrame(
        {
            "High": quote.get("high") or [],
            "Low": quote.get("low") or [],
            "Close": quote.get("close") or [],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
    )
    frame = frame.astype("float64").dropna(how="any")
    if frame.empty:
        raise ValueError(f"{ticker}: no OHLC prices returned")
    return frame.sort_index()


def _to_unix_timestamp(value: date | str, end_of_day: bool) -> int:
    parsed = pd.to_datetime(value).date()
    clock = time.max if end_of_day else time.min
    return int(datetime.combine(parsed, clock, tzinfo=timezone.utc).timestamp())


def latest_prices(prices: pd.DataFrame) -> pd.Series:
    """Return the most recent non-null price for each ticker."""
    if prices.empty:
        raise ValueError("Price data is empty")
    latest = prices.ffill().iloc[-1].dropna()
    if latest.empty:
        raise ValueError("No latest prices available")
    return latest
