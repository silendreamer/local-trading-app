from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_app.scanner2.config import MARKET_TIMEZONE, Scanner2Config, load_config
from trading_app.scanner2.polygon_client import PolygonRestClient, PolygonResult
from trading_app.scanner2.snapshot_store import load_snapshot, nearest_snapshot_time


LOGGER = logging.getLogger(__name__)
PREMARKET_START = time(4, 0)
PREMARKET_END = time(9, 30)
REGULAR_VOLUME_START = time(9, 30)


@dataclass
class ScanState:
    config: Scanner2Config
    client: PolygonRestClient
    run_date: date
    previous_day_stats: dict[str, dict[str, Any]]
    results_by_time: dict[str, dict[str, dict[str, Any]]]
    snapshots_by_time: dict[str, dict[str, Any]]
    snapshot_dir: Path | None = None


def get_previous_trading_day(run_date: date | None = None, client: PolygonRestClient | None = None) -> date:
    """Return the latest weekday before run_date. If a client is provided, skip dates with no grouped bars."""
    current = (run_date or datetime.now(MARKET_TIMEZONE).date()) - timedelta(days=1)
    for _ in range(10):
        if current.weekday() >= 5:
            current -= timedelta(days=1)
            continue
        if client is None:
            return current
        grouped = client.get_grouped_daily_bars(current.isoformat())
        if grouped.ok and (grouped.data or {}).get("results"):
            return current
        current -= timedelta(days=1)
    raise ValueError("Unable to find previous trading day")


def get_previous_day_stats(ticker: str, stats_by_ticker: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Get prior regular-session close and volume for a ticker."""
    stats = stats_by_ticker.get(ticker)
    if not stats:
        raise ValueError("Missing previous day stats")
    return stats


def get_current_price_at_or_before(ticker: str, scan_time: datetime, bars: pd.DataFrame, fallback_price: float | None) -> float:
    """Return latest bar close at or before scan_time, falling back to snapshot/latest price."""
    eligible = bars[bars.index <= scan_time] if not bars.empty else pd.DataFrame()
    if not eligible.empty and "close" in eligible:
        return float(eligible["close"].dropna().iloc[-1])
    if fallback_price is not None:
        return float(fallback_price)
    raise ValueError(f"{ticker}: missing current price")


def get_premarket_cumulative_volume(ticker: str, scan_time: datetime, bars: pd.DataFrame) -> int:
    """Calculate premarket volume from 4:00 AM ET through scan_time, capped at 9:30 AM."""
    del ticker
    end_time = min(scan_time, market_datetime(scan_time.date(), PREMARKET_END))
    premarket = bars[(bars.index >= market_datetime(scan_time.date(), PREMARKET_START)) & (bars.index <= end_time)]
    if premarket.empty:
        return 0
    return int(premarket["volume"].fillna(0).sum())


def get_premarket_high(ticker: str, scan_time: datetime, bars: pd.DataFrame) -> float | None:
    """Calculate premarket high from 4:00 AM ET through scan_time, capped at 9:30 AM."""
    del ticker
    end_time = min(scan_time, market_datetime(scan_time.date(), PREMARKET_END))
    premarket = bars[(bars.index >= market_datetime(scan_time.date(), PREMARKET_START)) & (bars.index <= end_time)]
    if premarket.empty:
        return None
    return float(premarket["high"].max())


def calculate_gap_pct(current_price: float, previous_close: float) -> float:
    """Calculate price change percentage from previous close."""
    if previous_close <= 0:
        raise ValueError("Previous close must be positive")
    return (current_price - previous_close) / previous_close * 100.0


def qualifies(row: dict[str, Any], config: Scanner2Config) -> bool:
    """Apply the momentum qualification rules."""
    if row.get("error"):
        return False
    current_price = value_or_none(row.get("current_price"))
    previous_close = value_or_none(row.get("prev_close_4pm"))
    previous_volume = value_or_none(row.get("prev_day_volume"))
    premarket_volume = value_or_none(row.get("premarket_volume"))
    if None in {current_price, previous_close, previous_volume, premarket_volume}:
        return False
    return (
        current_price >= previous_close * (1 + config.min_gap_pct / 100.0)
        and config.min_price <= current_price <= config.max_price
        and previous_volume >= config.min_prev_day_volume
        and premarket_volume >= config.min_premarket_volume
        and premarket_volume >= previous_volume * config.premarket_volume_to_prev_day_ratio
    )


def scan_initial_universe(scan_time: datetime, state: ScanState) -> list[str]:
    """At 8:00 AM, scan broad snapshot universe and keep top configured qualified candidates."""
    snapshot_data = snapshot_for_scan_time(scan_time, state)
    candidates = []
    for item in snapshot_items(snapshot_data):
        ticker = str(item.get("ticker", "")).upper()
        if not ticker:
            continue
        row = evaluate_ticker(ticker, scan_time, state, snapshot_item=item)
        state.results_by_time.setdefault(time_key(scan_time), {})[ticker] = row
        if row.get("qualified"):
            candidates.append(ticker)
    candidates = sorted(
        candidates,
        key=lambda symbol: state.results_by_time[time_key(scan_time)][symbol].get("change_pct", float("-inf")),
        reverse=True,
    )[: state.config.top_n]
    return candidates


def rescan_candidates(previous_candidates: list[str], scan_time: datetime, state: ScanState) -> list[str]:
    """At later scan times, rescan only the previous candidate list and keep narrowing."""
    kept = []
    snapshot_data = snapshot_for_scan_time(scan_time, state)
    snapshot_by_ticker = {
        str(item.get("ticker", "")).upper(): item
        for item in snapshot_items(snapshot_data)
    }
    for ticker in previous_candidates:
        row = evaluate_ticker(ticker, scan_time, state, snapshot_item=snapshot_by_ticker.get(ticker))
        state.results_by_time.setdefault(time_key(scan_time), {})[ticker] = row
        if row.get("qualified"):
            kept.append(ticker)
    return kept


def run_full_scan(
    config: Scanner2Config | None = None,
    client: PolygonRestClient | None = None,
    run_date: date | None = None,
    snapshot_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the complete staged scanner and return the final watchlist DataFrame."""
    from trading_app.scanner2.output_builder import build_final_dataframe

    scan_config = config or load_config()
    polygon = client or PolygonRestClient(
        scan_config.polygon_api_key,
        request_sleep_seconds=scan_config.request_sleep_seconds,
    )
    target_date = run_date or datetime.now(MARKET_TIMEZONE).date()
    previous_stats = load_previous_day_stats_from_snapshots(target_date, scan_config, snapshot_dir)
    if not previous_stats:
        previous_day = get_previous_trading_day(target_date, polygon)
        previous_stats = load_previous_day_stats(polygon, previous_day)
    state = ScanState(
        config=scan_config,
        client=polygon,
        run_date=target_date,
        previous_day_stats=previous_stats,
        results_by_time={},
        snapshots_by_time={},
        snapshot_dir=snapshot_dir,
    )

    candidates: list[str] = []
    for index, scan_time_text in enumerate(scan_config.scan_times):
        scan_time = scan_datetime(target_date, scan_time_text)
        LOGGER.info("scanner2 scan_time=%s previous_candidates=%s", scan_time_text, len(candidates))
        if index == 0:
            candidates = scan_initial_universe(scan_time, state)
        else:
            candidates = rescan_candidates(candidates, scan_time, state)
    return build_final_dataframe(state.results_by_time, previous_stats, scan_config)


def evaluate_ticker(
    ticker: str,
    scan_time: datetime,
    state: ScanState,
    snapshot_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one ticker at one scan time without raising ticker-level failures."""
    try:
        snapshot_stats = snapshot_previous_stats(snapshot_item)
        previous = snapshot_stats or get_previous_day_stats(ticker, state.previous_day_stats)
        bars = intraday_bars_for_ticker(state.client, ticker, state.run_date, scan_time)
        fallback_price = snapshot_price(snapshot_item)
        current_price = get_current_price_at_or_before(ticker, scan_time, bars, fallback_price)
        premarket_volume = snapshot_volume(snapshot_item)
        if premarket_volume is None:
            premarket_volume = get_premarket_cumulative_volume(ticker, scan_time, bars)
        premarket_high = get_premarket_high(ticker, scan_time, bars)
        change_pct = calculate_gap_pct(current_price, float(previous["prev_close_4pm"]))
        regular_volume = regular_session_volume_to_945(scan_time, bars)
        row = {
            "ticker": ticker,
            "scan_time": time_key(scan_time),
            "prev_close_4pm": float(previous["prev_close_4pm"]),
            "prev_day_volume": int(previous["prev_day_volume"]),
            "current_price": current_price,
            "premarket_volume": premarket_volume,
            "premarket_high": premarket_high,
            "change_pct": change_pct,
            "volume_9_30_to_9_45": regular_volume,
            "error": "",
        }
        row["qualified"] = qualifies(row, state.config)
        return row
    except Exception as exc:
        LOGGER.warning("scanner2 ticker failed ticker=%s scan_time=%s error=%s", ticker, scan_time, exc)
        return {"ticker": ticker, "scan_time": time_key(scan_time), "qualified": False, "error": str(exc)}


def load_previous_day_stats(client: PolygonRestClient, previous_day: date) -> dict[str, dict[str, Any]]:
    grouped = client.get_grouped_daily_bars(previous_day.isoformat())
    if not grouped.ok:
        raise ValueError(grouped.error)
    stats = {}
    for item in (grouped.data or {}).get("results") or []:
        ticker = str(item.get("T", "")).upper()
        if ticker:
            stats[ticker] = {
                "prev_close_4pm": float(item.get("c", 0.0)),
                "prev_day_volume": int(item.get("v", 0)),
            }
    if not stats:
        raise ValueError("No previous day grouped bars returned")
    return stats


def load_previous_day_stats_from_snapshots(
    run_date: date,
    config: Scanner2Config,
    snapshot_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build previous-day stats from persisted snapshot prevDay fields."""
    stats: dict[str, dict[str, Any]] = {}
    for scan_time_text in config.scan_times:
        scan_time = scan_datetime(run_date, scan_time_text)
        try:
            payload = load_snapshot(scan_time, snapshot_dir)
        except FileNotFoundError:
            continue
        for item in snapshot_items(payload):
            ticker = str(item.get("ticker", "")).upper()
            snapshot_stats = snapshot_previous_stats(item)
            if ticker and snapshot_stats:
                stats[ticker] = snapshot_stats
        if stats:
            return stats
    return stats


def intraday_bars_for_ticker(client: PolygonRestClient, ticker: str, run_date: date, scan_time: datetime) -> pd.DataFrame:
    start = market_datetime(run_date, PREMARKET_START)
    result = client.get_intraday_minute_bars(ticker, start, scan_time)
    if not result.ok:
        raise ValueError(result.error)
    return polygon_bars_to_frame((result.data or {}).get("results") or [])


def polygon_bars_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(item["t"], unit="ms", utc=True).tz_convert(MARKET_TIMEZONE),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v", 0),
            }
            for item in results
            if "t" in item
        ]
    )
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return frame.set_index("timestamp").sort_index()


def latest_price(client: PolygonRestClient, ticker: str) -> float | None:
    result = client.get_latest_price(ticker)
    if not result.ok:
        return None
    payload = result.data or {}
    return value_or_none((payload.get("results") or {}).get("p"))


def snapshot_price(snapshot_item: dict[str, Any] | None) -> float | None:
    if not snapshot_item:
        return None
    last_trade = snapshot_item.get("lastTrade") or {}
    minute = snapshot_item.get("min") or {}
    day = snapshot_item.get("day") or {}
    for value in [last_trade.get("p"), minute.get("c"), day.get("c")]:
        parsed = value_or_none(value)
        if parsed is not None:
            return parsed
    return None


def snapshot_volume(snapshot_item: dict[str, Any] | None) -> int | None:
    if not snapshot_item:
        return None
    day = snapshot_item.get("day") or {}
    parsed = value_or_none(day.get("v"))
    return int(parsed) if parsed is not None else None


def snapshot_previous_stats(snapshot_item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot_item:
        return None
    previous_day = snapshot_item.get("prevDay") or {}
    previous_close = value_or_none(previous_day.get("c"))
    previous_volume = value_or_none(previous_day.get("v"))
    if previous_close is None:
        return None
    return {
        "prev_close_4pm": previous_close,
        "prev_day_volume": int(previous_volume or 0),
    }


def snapshot_for_scan_time(scan_time: datetime, state: ScanState) -> dict[str, Any]:
    key = time_key(scan_time)
    if key in state.snapshots_by_time:
        return state.snapshots_by_time[key]
    snapshot_time = nearest_snapshot_time(scan_time, state.snapshot_dir)
    payload = load_snapshot(snapshot_time, state.snapshot_dir)
    state.snapshots_by_time[key] = payload
    return payload


def snapshot_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tickers = payload.get("tickers")
    if isinstance(tickers, list):
        return tickers
    results = payload.get("results")
    if isinstance(results, list):
        return results
    return []


def regular_session_volume_to_945(scan_time: datetime, bars: pd.DataFrame) -> int | None:
    if scan_time.time() < time(9, 45):
        return None
    start = market_datetime(scan_time.date(), REGULAR_VOLUME_START)
    end = market_datetime(scan_time.date(), time(9, 45))
    regular = bars[(bars.index >= start) & (bars.index <= end)]
    return int(regular["volume"].fillna(0).sum()) if not regular.empty else 0


def scan_datetime(run_date: date, scan_time_text: str) -> datetime:
    hour, minute = [int(part) for part in scan_time_text.split(":")]
    return market_datetime(run_date, time(hour, minute))


def market_datetime(run_date: date, clock: time) -> datetime:
    return datetime.combine(run_date, clock, tzinfo=MARKET_TIMEZONE)


def time_key(scan_time: datetime) -> str:
    return scan_time.strftime("%H_%M")


def value_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    parsed = float(value)
    return parsed if pd.notna(parsed) else None
