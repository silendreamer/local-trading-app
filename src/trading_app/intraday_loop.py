from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import logging
from threading import Event, Lock, Thread
from typing import Callable
from zoneinfo import ZoneInfo


LOGGER = logging.getLogger("trading_app.scheduler")
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
DEFAULT_CHECK_INTERVAL = timedelta(minutes=15)

TickerCheck = Callable[[list[str]], int]


@dataclass(frozen=True)
class TradingLoopSnapshot:
    running: bool
    status: str
    next_check_time: datetime | None
    last_check_time: datetime | None
    last_message: str
    last_error: str
    checked_ticker_count: int
    dry_run: bool
    alpaca_paper: bool


@dataclass
class _TradingLoopState:
    running: bool = False
    status: str = "stopped"
    next_check_time: datetime | None = None
    last_check_time: datetime | None = None
    last_message: str = "Trading loop is stopped."
    last_error: str = ""
    checked_ticker_count: int = 0
    dry_run: bool = True
    alpaca_paper: bool = True


class TradingLoopManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._state = _TradingLoopState()

    def start(
        self,
        tickers: list[str],
        *,
        dry_run: bool,
        alpaca_paper: bool,
        check_tickers: TickerCheck | None = None,
        interval: timedelta = DEFAULT_CHECK_INTERVAL,
        now: datetime | None = None,
    ) -> str:
        current_time = market_now(now)
        if not is_market_open(current_time):
            LOGGER.info("Scheduler start rejected outside market hours at %s", format_market_time(current_time))
            self._set_stopped(
                f"Trading loop was not started because the market is closed. Next open: "
                f"{format_market_time(next_market_open(current_time))}."
            )
            return self.snapshot().last_message

        with self._lock:
            if self._state.running and (self._thread is None or self._scheduler_is_alive()):
                LOGGER.info("Scheduler start ignored because scheduler is already running")
                return "Trading loop is already running."
            if self._state.running:
                LOGGER.warning("Scheduler state was running but background thread was not alive; resetting state")

            self._stop_event.clear()
            self._state = _TradingLoopState(
                running=True,
                status="running",
                next_check_time=current_time,
                last_check_time=None,
                last_message="Trading loop started in paper/DRY_RUN control mode.",
                dry_run=dry_run,
                alpaca_paper=alpaca_paper,
            )

        LOGGER.info(
            "Scheduler start: tickers=%s interval_seconds=%s dry_run=%s alpaca_paper=%s",
            len(tickers),
            int(interval.total_seconds()),
            dry_run,
            alpaca_paper,
        )
        self._thread = Thread(
            target=self._run_loop,
            args=(list(tickers), check_tickers or _count_tickers, interval),
            name="intraday-trading-loop",
            daemon=True,
        )
        self._thread.start()
        return "Trading loop started."

    def stop(self) -> str:
        self._stop_event.set()
        self._set_stopped("Trading loop stopped manually.")
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        LOGGER.info("Scheduler stop requested manually")
        return "Trading loop stopped manually."

    def snapshot(self) -> TradingLoopSnapshot:
        with self._lock:
            return TradingLoopSnapshot(**self._state.__dict__)

    def _run_loop(
        self,
        tickers: list[str],
        check_tickers: TickerCheck,
        interval: timedelta,
    ) -> None:
        while not self._stop_event.is_set():
            current_time = market_now()
            if current_time >= market_close_for(current_time):
                LOGGER.info("Scheduler stop: reached market close at %s", format_market_time(current_time))
                self._set_stopped("Trading loop stopped automatically at the 4:00 PM market close.")
                return
            if not is_market_open(current_time):
                LOGGER.info("Scheduler stop: market closed at %s", format_market_time(current_time))
                self._set_stopped("Trading loop stopped because the market is closed.")
                return

            try:
                LOGGER.info("Scheduler scan cycle start: %s tickers at %s", len(tickers), format_market_time(current_time))
                checked_count = check_tickers(tickers)
                last_error = ""
                last_message = f"Checked {checked_count} configured tickers."
                LOGGER.info("Scheduler scan cycle end: checked_count=%s", checked_count)
            except Exception as exc:
                checked_count = 0
                last_error = str(exc)
                last_message = "Ticker check failed; loop remains in DRY_RUN control mode."
                LOGGER.exception("Scheduler scan cycle error: %s", exc)

            next_check = min(current_time + interval, market_close_for(current_time))
            with self._lock:
                self._state.last_check_time = current_time
                self._state.next_check_time = next_check
                self._state.last_message = last_message
                self._state.last_error = last_error
                self._state.checked_ticker_count = checked_count

            wait_seconds = max((next_check - market_now()).total_seconds(), 0.0)
            if self._stop_event.wait(wait_seconds):
                self._set_stopped("Trading loop stopped manually.")
                LOGGER.info("Scheduler stop: manual stop received while waiting")
                return

        self._set_stopped("Trading loop stopped manually.")
        LOGGER.info("Scheduler stop: stop event set")

    def _set_stopped(self, message: str) -> None:
        with self._lock:
            self._state.running = False
            self._state.status = "stopped"
            self._state.next_check_time = None
            self._state.last_message = message

    def _scheduler_is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


def market_now(now: datetime | None = None) -> datetime:
    current_time = now or datetime.now(tz=MARKET_TIMEZONE)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current_time.astimezone(MARKET_TIMEZONE)


def is_market_open(now: datetime | None = None) -> bool:
    current_time = market_now(now)
    if current_time.weekday() >= 5:
        return False
    return MARKET_OPEN <= current_time.time() < MARKET_CLOSE


def market_close_for(now: datetime) -> datetime:
    current_time = market_now(now)
    return current_time.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0,
    )


def next_market_open(now: datetime | None = None) -> datetime:
    current_time = market_now(now)
    candidate = current_time.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0,
    )
    if current_time.time() < MARKET_OPEN and current_time.weekday() < 5:
        return candidate
    candidate = candidate + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def format_market_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    return market_now(value).strftime("%Y-%m-%d %I:%M:%S %p %Z")


def _count_tickers(tickers: list[str]) -> int:
    return len(tickers)
