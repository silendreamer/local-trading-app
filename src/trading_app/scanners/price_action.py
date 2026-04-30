from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging

import pandas as pd

from trading_app.data import MarketDataRequest, fetch_ohlc
from trading_app.intraday_loop import format_market_time, is_market_open, market_now


LOGGER = logging.getLogger("trading_app.scanners.price_action")
FRESHNESS_LIMIT = timedelta(minutes=30)


@dataclass(frozen=True)
class PriceActionScanResult:
    ticker: str
    current_price: float | None
    signal: str
    reason: str
    timestamp: str
    data_freshness_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scan_price_action(
    tickers: list[str],
    *,
    now: datetime | None = None,
    interval: str = "15m",
    freshness_limit: timedelta = FRESHNESS_LIMIT,
    fetcher=fetch_ohlc,
) -> list[PriceActionScanResult]:
    """Scan configured tickers with recent intraday candles.

    Missing, stale, or market-closed rows are returned as SKIP results and logged.
    The scanner is signal-only; it does not place orders.
    """
    current_time = market_now(now)
    if not tickers:
        raise ValueError("At least one ticker is required")

    if not is_market_open(current_time):
        return [
            _skip_result(
                ticker,
                reason="Market is closed",
                timestamp=current_time,
                freshness_status="market_closed",
            )
            for ticker in tickers
        ]

    try:
        ohlc_by_ticker = fetcher(
            MarketDataRequest(
                tickers=tickers,
                start=current_time.date() - timedelta(days=5),
                end=current_time.date() + timedelta(days=1),
                interval=interval,
            )
        )
    except Exception as exc:
        LOGGER.warning("Price action scan failed for all tickers: %s", exc)
        return [
            _skip_result(
                ticker,
                reason=f"Market data fetch failed: {exc}",
                timestamp=current_time,
                freshness_status="missing",
            )
            for ticker in tickers
        ]

    results: list[PriceActionScanResult] = []
    for ticker in tickers:
        frame = ohlc_by_ticker.get(ticker)
        if frame is None or frame.empty:
            results.append(
                _skip_result(
                    ticker,
                    reason="Missing intraday candles",
                    timestamp=current_time,
                    freshness_status="missing",
                )
            )
            continue

        clean_frame = frame.loc[:, ["High", "Low", "Close"]].dropna().sort_index()
        if clean_frame.empty:
            results.append(
                _skip_result(
                    ticker,
                    reason="Intraday candles contain no complete OHLC rows",
                    timestamp=current_time,
                    freshness_status="missing",
                )
            )
            continue

        latest_timestamp = _as_market_time(clean_frame.index[-1])
        if latest_timestamp is None:
            results.append(
                _skip_result(
                    ticker,
                    reason="Latest candle timestamp is invalid",
                    timestamp=current_time,
                    freshness_status="missing",
                )
            )
            continue

        age = current_time - latest_timestamp
        if age < timedelta(0) or age > freshness_limit:
            results.append(
                _skip_result(
                    ticker,
                    reason=f"Latest 15-minute candle is stale: {format_market_time(latest_timestamp)}",
                    timestamp=latest_timestamp,
                    freshness_status="stale",
                )
            )
            continue

        results.append(_build_signal(ticker, clean_frame, latest_timestamp))

    return results


def results_to_dataframe(results: list[PriceActionScanResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict() for result in results])


def _build_signal(
    ticker: str,
    frame: pd.DataFrame,
    latest_timestamp: datetime,
) -> PriceActionScanResult:
    latest = frame.iloc[-1]
    current_price = float(latest["Close"])
    if len(frame) < 4:
        return PriceActionScanResult(
            ticker=ticker,
            current_price=current_price,
            signal="HOLD",
            reason="Not enough recent intraday candles for price action confirmation",
            timestamp=format_market_time(latest_timestamp),
            data_freshness_status="fresh",
        )

    prior = frame.iloc[-4:-1]
    prior_high = float(prior["High"].max())
    prior_low = float(prior["Low"].min())
    previous_close = float(frame["Close"].iloc[-2])

    if current_price > prior_high:
        signal = "BUY_WATCH"
        reason = "Latest close broke above the prior three-candle high"
    elif current_price < prior_low:
        signal = "SELL_WATCH"
        reason = "Latest close broke below the prior three-candle low"
    elif current_price > previous_close:
        signal = "HOLD_BULLISH"
        reason = "Latest close is rising but has not broken recent resistance"
    elif current_price < previous_close:
        signal = "HOLD_BEARISH"
        reason = "Latest close is falling but has not broken recent support"
    else:
        signal = "HOLD"
        reason = "Latest close is unchanged"

    return PriceActionScanResult(
        ticker=ticker,
        current_price=current_price,
        signal=signal,
        reason=reason,
        timestamp=format_market_time(latest_timestamp),
        data_freshness_status="fresh",
    )


def _skip_result(
    ticker: str,
    *,
    reason: str,
    timestamp: datetime,
    freshness_status: str,
) -> PriceActionScanResult:
    LOGGER.info("Skipping %s in price action scan: %s", ticker, reason)
    return PriceActionScanResult(
        ticker=ticker,
        current_price=None,
        signal="SKIP",
        reason=reason,
        timestamp=format_market_time(timestamp),
        data_freshness_status=freshness_status,
    )


def _as_market_time(value: object) -> datetime | None:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.to_pydatetime().astimezone(market_now().tzinfo)
