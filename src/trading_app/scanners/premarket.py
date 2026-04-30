from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


MARKET_TIMEZONE = ZoneInfo("America/New_York")
PREMARKET_START = time(4, 0)
REGULAR_MARKET_START = time(9, 30)


@dataclass(frozen=True)
class ScannerRow:
    ticker: str
    previous_day_high: float | None = None
    previous_day_close: float | None = None
    previous_day_volume: int | None = None
    pre_market_high: float | None = None
    pre_market_volume: int | None = None
    current_price: float | None = None
    percent_gap_from_previous_close: float | None = None
    percent_move_from_previous_day_high: float | None = None
    above_pre_market_high: bool | None = None
    latest_price_timestamp: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "previous_day_high": self.previous_day_high,
            "previous_day_close": self.previous_day_close,
            "previous_day_volume": self.previous_day_volume,
            "pre_market_high": self.pre_market_high,
            "pre_market_volume": self.pre_market_volume,
            "current_price": self.current_price,
            "percent_gap_from_previous_close": self.percent_gap_from_previous_close,
            "percent_move_from_previous_day_high": self.percent_move_from_previous_day_high,
            "above_pre_market_high": self.above_pre_market_high,
            "latest_price_timestamp": self.latest_price_timestamp,
            "error": self.error,
        }


@dataclass(frozen=True)
class MomentumScannerConfig:
    gap_threshold_percent: float = 5.0
    min_premarket_volume: int = 100_000
    min_price: float = 2.0
    max_price: float = 20.0
    top_n: int = 20
    chunk_size: int = 100


def normalize_tickers(raw_tickers: list[str] | str) -> list[str]:
    if isinstance(raw_tickers, str):
        parts = raw_tickers.replace("\n", ",").split(",")
    else:
        parts = raw_tickers
    normalized = []
    seen = set()
    for ticker in parts:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def scan_momentum_candidates(
    tickers: list[str],
    config: MomentumScannerConfig | None = None,
    now: datetime | None = None,
    daily_fetcher=None,
    intraday_fetcher=None,
) -> pd.DataFrame:
    scan_config = config or MomentumScannerConfig()
    current_time = normalize_market_time(now)
    symbols = normalize_tickers(tickers)
    if not symbols:
        return pd.DataFrame(columns=momentum_candidate_columns())

    daily_by_ticker = (daily_fetcher or fetch_daily_history_batch)(symbols)
    intraday_by_ticker = (intraday_fetcher or fetch_intraday_history_batch)(symbols)
    candidates = []
    errors = []
    for ticker in symbols:
        row = build_momentum_candidate_row(
            ticker,
            current_time,
            daily_by_ticker.get(ticker, pd.DataFrame()),
            intraday_by_ticker.get(ticker, pd.DataFrame()),
        )
        if row.get("error"):
            errors.append(row)
            continue
        if not passes_momentum_filters(row, scan_config):
            continue
        candidates.append(row)

    candidates = sorted(
        candidates,
        key=lambda row: (
            row.get("percent_gap_from_previous_close") or float("-inf"),
            row.get("relative_volume") or float("-inf"),
        ),
        reverse=True,
    )[: scan_config.top_n]
    return pd.DataFrame(candidates + errors, columns=momentum_candidate_columns())


def momentum_candidate_columns() -> list[str]:
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


def build_momentum_candidate_row(
    ticker: str,
    now: datetime,
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
) -> dict:
    try:
        previous_day = previous_trading_day_row(daily, now)
        intraday = normalize_intraday_index(intraday)
        today_intraday = intraday[intraday.index.date == now.date()]
        current_price = latest_close(today_intraday)
        latest_timestamp = latest_timestamp_text(today_intraday)
        premarket = premarket_candles(today_intraday)
        if premarket.empty:
            raise ValueError("No premarket candles for today")

        previous_high = float(previous_day["High"])
        previous_close = float(previous_day["Close"])
        previous_volume = int(previous_day["Volume"])
        premarket_high = latest_or_none(premarket["High"].max())
        premarket_volume = int(premarket["Volume"].sum()) if "Volume" in premarket else 0
        signal, reason = continuation_signal(
            current_price=current_price,
            previous_day_high=previous_high,
            premarket_high=premarket_high,
        )
        return {
            "ticker": ticker,
            "previous_day_high": previous_high,
            "previous_day_close": previous_close,
            "previous_day_volume": previous_volume,
            "pre_market_high": premarket_high,
            "pre_market_volume": premarket_volume,
            "current_price": current_price,
            "percent_gap_from_previous_close": percent_change(current_price, previous_close),
            "percent_move_from_previous_day_high": percent_change(current_price, previous_high),
            "relative_volume": premarket_volume / previous_volume if previous_volume else None,
            "above_pre_market_high": current_price > premarket_high if premarket_high is not None else None,
            "latest_price_timestamp": latest_timestamp,
            "continuation_signal": signal,
            "continuation_reason": reason,
            "error": "",
        }
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def passes_momentum_filters(row: dict, config: MomentumScannerConfig) -> bool:
    gap = row.get("percent_gap_from_previous_close")
    volume = row.get("pre_market_volume")
    price = row.get("current_price")
    if gap is None or volume is None or price is None:
        return False
    return (
        gap > config.gap_threshold_percent
        and volume > config.min_premarket_volume
        and config.min_price <= price <= config.max_price
    )


def continuation_signal(
    current_price: float,
    previous_day_high: float,
    premarket_high: float | None,
) -> tuple[str, str]:
    if premarket_high is not None and current_price > premarket_high and current_price > previous_day_high:
        return "BREAKOUT_CONTINUATION", "Current price is above both premarket high and previous day high"
    if premarket_high is not None and current_price > premarket_high:
        return "PREMARKET_HIGH_BREAK", "Current price is above premarket high"
    if current_price > previous_day_high:
        return "PREVIOUS_HIGH_BREAK", "Current price is above previous day high"
    return "WATCH", "Current price has not cleared premarket or previous day high"


def fetch_daily_history_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return fetch_yfinance_batch(tickers, period="15d", interval="1d", prepost=False)


def fetch_intraday_history_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return fetch_yfinance_batch(tickers, period="1d", interval="1m", prepost=True)


def fetch_yfinance_batch(tickers: list[str], period: str, interval: str, prepost: bool) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for chunk in chunked(normalize_tickers(tickers), 100):
        yfinance_symbols = [to_yfinance_symbol(ticker) for ticker in chunk]
        raw = yf.download(
            tickers=yfinance_symbols,
            period=period,
            interval=interval,
            prepost=prepost,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        for ticker, yfinance_symbol in zip(chunk, yfinance_symbols):
            result[ticker] = extract_ticker_frame(raw, yfinance_symbol, len(chunk) == 1)
    return result


def extract_ticker_frame(raw: pd.DataFrame, yfinance_symbol: str, single_ticker: bool) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if single_ticker and not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy()
    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()
    if yfinance_symbol in raw.columns.get_level_values(0):
        return raw[yfinance_symbol].copy()
    if yfinance_symbol in raw.columns.get_level_values(1):
        return raw.xs(yfinance_symbol, axis=1, level=1).copy()
    return pd.DataFrame()


def chunked(values: list[str], size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def to_yfinance_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def normalize_market_time(now: datetime | None = None) -> datetime:
    current_time = now or datetime.now(MARKET_TIMEZONE)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=MARKET_TIMEZONE)
    return current_time.astimezone(MARKET_TIMEZONE)


def scan_premarket(
    tickers: list[str],
    now: datetime | None = None,
    daily_fetcher=None,
    intraday_fetcher=None,
) -> pd.DataFrame:
    current_time = normalize_market_time(now)
    rows = [
        scan_ticker(
            ticker,
            current_time,
            daily_fetcher=daily_fetcher or fetch_daily_history,
            intraday_fetcher=intraday_fetcher or fetch_intraday_history,
        ).to_dict()
        for ticker in normalize_tickers(tickers)
    ]
    return pd.DataFrame(rows, columns=scanner_columns())


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
        "above_pre_market_high",
        "latest_price_timestamp",
        "error",
    ]


def scan_ticker(ticker: str, now: datetime, daily_fetcher, intraday_fetcher) -> ScannerRow:
    try:
        daily = daily_fetcher(ticker)
        previous_day = previous_trading_day_row(daily, now)
        intraday = intraday_fetcher(ticker)
        intraday = normalize_intraday_index(intraday)

        today_intraday = intraday[intraday.index.date == now.date()]
        current_price = latest_close(today_intraday)
        latest_timestamp = latest_timestamp_text(today_intraday)
        premarket = premarket_candles(today_intraday)

        previous_high = float(previous_day["High"])
        previous_close = float(previous_day["Close"])
        premarket_high = latest_or_none(premarket["High"].max()) if not premarket.empty else None
        premarket_volume = int(premarket["Volume"].sum()) if not premarket.empty and "Volume" in premarket else None

        return ScannerRow(
            ticker=ticker,
            previous_day_high=previous_high,
            previous_day_close=previous_close,
            previous_day_volume=int(previous_day["Volume"]),
            pre_market_high=premarket_high,
            pre_market_volume=premarket_volume,
            current_price=current_price,
            percent_gap_from_previous_close=percent_change(current_price, previous_close),
            percent_move_from_previous_day_high=percent_change(current_price, previous_high),
            above_pre_market_high=current_price > premarket_high if premarket_high is not None else None,
            latest_price_timestamp=latest_timestamp,
            error="" if premarket_high is not None else "No premarket candles for today",
        )
    except Exception as exc:
        return ScannerRow(ticker=ticker, error=str(exc))


def fetch_daily_history(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="15d", interval="1d", auto_adjust=False)


def fetch_intraday_history(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True, auto_adjust=False)


def previous_trading_day_row(daily: pd.DataFrame, now: datetime) -> pd.Series:
    if daily.empty:
        raise ValueError("No daily history returned")
    required = {"High", "Close", "Volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Daily history missing columns: {', '.join(sorted(missing))}")

    normalized = daily.dropna(subset=["High", "Close", "Volume"]).copy()
    normalized.index = pd.to_datetime(normalized.index)
    previous_days = normalized[normalized.index.date < now.date()]
    if previous_days.empty:
        raise ValueError("No previous trading day found")
    return previous_days.sort_index().iloc[-1]


def normalize_intraday_index(intraday: pd.DataFrame) -> pd.DataFrame:
    if intraday.empty:
        raise ValueError("No intraday history returned")
    if "Close" not in intraday.columns:
        raise ValueError("Intraday history missing Close column")

    normalized = intraday.copy()
    index = pd.to_datetime(normalized.index)
    if index.tz is None:
        index = index.tz_localize(MARKET_TIMEZONE)
    else:
        index = index.tz_convert(MARKET_TIMEZONE)
    normalized.index = index
    return normalized.sort_index()


def premarket_candles(intraday: pd.DataFrame) -> pd.DataFrame:
    if intraday.empty:
        return intraday
    clock = pd.Series(intraday.index.time, index=intraday.index)
    return intraday[(clock >= PREMARKET_START) & (clock < REGULAR_MARKET_START)]


def latest_close(intraday: pd.DataFrame) -> float:
    if intraday.empty:
        raise ValueError("No intraday candles for today")
    closes = intraday["Close"].dropna()
    if closes.empty:
        raise ValueError("No latest intraday close available")
    return float(closes.iloc[-1])


def latest_timestamp_text(intraday: pd.DataFrame) -> str:
    closes = intraday["Close"].dropna()
    if closes.empty:
        return ""
    return str(closes.index[-1])


def latest_or_none(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def percent_change(current: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return (current - reference) / reference * 100.0
