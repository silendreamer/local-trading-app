from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading

from trading_app.scanner2.config import MARKET_TIMEZONE, Scanner2Config
from trading_app.scanner2.snapshot_store import run_snapshot_service


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackgroundSnapshotStatus:
    enabled: bool
    running: bool
    interval_minutes: int
    poll_seconds: int
    started_at: datetime | None
    message: str


_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STARTED_AT: datetime | None = None
_MESSAGE = "Snapshot service has not started."


def start_background_snapshot_service(
    config: Scanner2Config,
    *,
    enabled: bool,
    interval_minutes: int = 15,
    poll_seconds: int = 30,
    retention_days: int = 3,
) -> BackgroundSnapshotStatus:
    """Start one daemon snapshot worker for the current Streamlit process."""
    global _MESSAGE, _STARTED_AT, _THREAD

    with _LOCK:
        if not enabled:
            _MESSAGE = "Automatic snapshots are disabled for this environment."
            return _status(enabled, interval_minutes, poll_seconds)

        if not config.polygon_api_key:
            _MESSAGE = "POLYGON_API_KEY is not configured; automatic snapshots are not running."
            return _status(enabled, interval_minutes, poll_seconds)

        if _THREAD and _THREAD.is_alive():
            return _status(enabled, interval_minutes, poll_seconds)

        _STARTED_AT = datetime.now(MARKET_TIMEZONE)
        _MESSAGE = "Automatic snapshot service is running."
        _THREAD = threading.Thread(
            target=_run_service,
            kwargs={
                "config": config,
                "interval_minutes": interval_minutes,
                "poll_seconds": poll_seconds,
                "retention_days": retention_days,
            },
            name="scanner2-snapshot-service",
            daemon=True,
        )
        _THREAD.start()
        return _status(enabled, interval_minutes, poll_seconds)


def background_snapshot_status(
    *,
    enabled: bool,
    interval_minutes: int = 15,
    poll_seconds: int = 30,
) -> BackgroundSnapshotStatus:
    with _LOCK:
        return _status(enabled, interval_minutes, poll_seconds)


def _run_service(config: Scanner2Config, interval_minutes: int, poll_seconds: int, retention_days: int) -> None:
    global _MESSAGE
    try:
        run_snapshot_service(
            config=config,
            interval_minutes=interval_minutes,
            poll_seconds=poll_seconds,
            retention_days=retention_days,
        )
    except Exception:
        LOGGER.exception("Background Scanner2 snapshot service stopped unexpectedly")
        with _LOCK:
            _MESSAGE = "Automatic snapshot service stopped unexpectedly. Refresh the app to restart it."


def _status(enabled: bool, interval_minutes: int, poll_seconds: int) -> BackgroundSnapshotStatus:
    return BackgroundSnapshotStatus(
        enabled=enabled,
        running=bool(_THREAD and _THREAD.is_alive()),
        interval_minutes=interval_minutes,
        poll_seconds=poll_seconds,
        started_at=_STARTED_AT,
        message=_MESSAGE,
    )
