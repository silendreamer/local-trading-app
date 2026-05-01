from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import logging
from pathlib import Path

from trading_app.config import PROJECT_ROOT
from trading_app.scanner2.config import MARKET_TIMEZONE, Scanner2Config, load_config
from trading_app.scanner2.github_snapshot_store import GitHubSnapshotStore, load_github_snapshot_config
from trading_app.scanner2.polygon_client import PolygonRestClient


LOGGER = logging.getLogger(__name__)
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_RETENTION_DAYS = 3


def snapshot_path(scan_time: datetime, snapshot_dir: Path | None = None) -> Path:
    """Return the persisted snapshot path for a market-time timestamp."""
    target_dir = snapshot_dir or SNAPSHOT_DIR
    return target_dir / f"snapshot_{scan_time.astimezone(MARKET_TIMEZONE):%Y%m%d_%H%M}.json"


def save_snapshot(payload: dict, scan_time: datetime, snapshot_dir: Path | None = None) -> Path:
    """Save a Polygon full-market snapshot response to disk."""
    path = snapshot_path(scan_time, snapshot_dir)
    github_store = github_snapshot_store()
    if snapshot_dir is None and github_store:
        github_store.save_text(
            path.name,
            json.dumps(payload),
            f"Save Polygon snapshot {path.name}",
        )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_snapshot(scan_time: datetime, snapshot_dir: Path | None = None) -> dict:
    """Load a persisted Polygon snapshot response for a scan time."""
    path = snapshot_path(scan_time, snapshot_dir)
    github_store = github_snapshot_store()
    if snapshot_dir is None and github_store:
        return json.loads(github_store.read_text(path.name))
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_exists(scan_time: datetime, snapshot_dir: Path | None = None) -> bool:
    path = snapshot_path(scan_time, snapshot_dir)
    github_store = github_snapshot_store()
    if snapshot_dir is None and github_store:
        return github_store.exists(path.name)
    return path.exists()


def snapshot_time_from_path(path: Path) -> datetime | None:
    """Parse a persisted snapshot filename timestamp."""
    stamp = path.stem.removeprefix("snapshot_")
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M").replace(tzinfo=MARKET_TIMEZONE)
    except ValueError:
        return None


def delete_snapshots_older_than(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    snapshot_dir: Path | None = None,
    now: datetime | None = None,
) -> int:
    """Delete persisted snapshot files older than the retention window."""
    target_dir = snapshot_dir or SNAPSHOT_DIR
    cutoff = (now or datetime.now(MARKET_TIMEZONE)).astimezone(MARKET_TIMEZONE) - timedelta(days=retention_days)
    github_store = github_snapshot_store()
    if snapshot_dir is None and github_store:
        deleted = 0
        for name in github_store.list_snapshot_names():
            snapshot_time = snapshot_time_from_path(Path(name))
            if snapshot_time is None or snapshot_time >= cutoff:
                continue
            try:
                if github_store.delete(name, f"Delete old Polygon snapshot {name}"):
                    deleted += 1
            except Exception:
                LOGGER.exception("Failed to delete old GitHub snapshot name=%s", name)
        return deleted

    if not target_dir.exists():
        return 0

    deleted = 0
    for path in target_dir.glob("snapshot_*.json"):
        snapshot_time = snapshot_time_from_path(path)
        if snapshot_time is None or snapshot_time >= cutoff:
            continue
        try:
            path.unlink()
            deleted += 1
        except OSError:
            LOGGER.exception("Failed to delete old snapshot path=%s", path)
    return deleted


def recent_snapshots(limit: int = 5, snapshot_dir: Path | None = None) -> list[tuple[datetime, Path]]:
    """Return the most recent persisted snapshots by encoded capture time."""
    target_dir = snapshot_dir or SNAPSHOT_DIR
    github_store = github_snapshot_store()
    if snapshot_dir is None and github_store:
        snapshots = []
        for name in github_store.list_snapshot_names():
            snapshot_time = snapshot_time_from_path(Path(name))
            if snapshot_time is not None:
                snapshots.append((snapshot_time, target_dir / name))
        return sorted(snapshots, key=lambda item: item[0], reverse=True)[:limit]

    if not target_dir.exists():
        return []

    snapshots = []
    for path in target_dir.glob("snapshot_*.json"):
        snapshot_time = snapshot_time_from_path(path)
        if snapshot_time is not None:
            snapshots.append((snapshot_time, path))
    return sorted(snapshots, key=lambda item: item[0], reverse=True)[:limit]


def latest_snapshot_time(snapshot_dir: Path | None = None) -> datetime | None:
    recent = recent_snapshots(1, snapshot_dir)
    if not recent:
        return None
    return recent[0][0]


def capture_snapshot(
    client: PolygonRestClient,
    scan_time: datetime | None = None,
    snapshot_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Call Polygon snapshot once and persist the full response."""
    target_time = (scan_time or datetime.now(MARKET_TIMEZONE)).astimezone(MARKET_TIMEZONE)
    path = snapshot_path(target_time, snapshot_dir)
    if path.exists() and not overwrite:
        LOGGER.info("Snapshot already exists path=%s", path)
        return path
    result = client.get_all_tickers_snapshot()
    if not result.ok:
        raise ValueError(result.error)
    return save_snapshot(result.data or {}, target_time, snapshot_dir)


def due_scan_time(now: datetime, interval_minutes: int = 15) -> datetime:
    """Floor current market time to the nearest interval."""
    current = now.astimezone(MARKET_TIMEZONE)
    minute = current.minute - (current.minute % interval_minutes)
    return current.replace(minute=minute, second=0, microsecond=0)


def is_interval_boundary(now: datetime, interval_minutes: int = 15) -> bool:
    """Return whether current market time is on an interval boundary minute."""
    current = now.astimezone(MARKET_TIMEZONE)
    return current.minute % interval_minutes == 0


def next_interval_boundary(now: datetime, interval_minutes: int = 15) -> datetime:
    """Return the next interval boundary after the current market time."""
    current = now.astimezone(MARKET_TIMEZONE).replace(second=0, microsecond=0)
    minutes_to_add = interval_minutes - (current.minute % interval_minutes)
    if minutes_to_add == 0:
        minutes_to_add = interval_minutes
    return current + timedelta(minutes=minutes_to_add)


def run_snapshot_service(
    config: Scanner2Config | None = None,
    snapshot_dir: Path | None = None,
    interval_minutes: int = 15,
    poll_seconds: int = 30,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    run_once: bool = False,
) -> None:
    """Persist Polygon snapshots on a fixed interval throughout the trading day."""
    scanner_config = config or load_config()
    client = PolygonRestClient(
        scanner_config.polygon_api_key,
        request_sleep_seconds=scanner_config.request_sleep_seconds,
    )
    while True:
        now = datetime.now(MARKET_TIMEZONE)
        target_time = due_scan_time(now, interval_minutes)
        delete_snapshots_older_than(retention_days, snapshot_dir=snapshot_dir, now=now)
        if is_interval_boundary(now, interval_minutes) and is_capture_window(target_time):
            try:
                capture_snapshot(client, target_time, snapshot_dir=snapshot_dir)
            except Exception:
                LOGGER.exception("Snapshot capture failed target_time=%s", target_time)
        if run_once:
            return
        import time as clock

        clock.sleep(poll_seconds)


def is_capture_window(scan_time: datetime) -> bool:
    """Limit captures to broad market day hours, including premarket."""
    market_clock = scan_time.astimezone(MARKET_TIMEZONE).time()
    return time(4, 0) <= market_clock <= time(20, 0) and scan_time.weekday() < 5


def available_snapshot_times(run_date: date, snapshot_dir: Path | None = None) -> list[datetime]:
    """List persisted snapshot timestamps for a date."""
    target_dir = snapshot_dir or SNAPSHOT_DIR
    if not target_dir.exists():
        return []
    prefix = f"snapshot_{run_date:%Y%m%d}_"
    times = []
    for path in target_dir.glob(f"{prefix}*.json"):
        stamp = path.stem.removeprefix("snapshot_")
        try:
            times.append(datetime.strptime(stamp, "%Y%m%d_%H%M").replace(tzinfo=MARKET_TIMEZONE))
        except ValueError:
            continue
    return sorted(times)


def nearest_snapshot_time(scan_time: datetime, snapshot_dir: Path | None = None, max_age_minutes: int = 15) -> datetime:
    """Find the nearest persisted snapshot at or before scan_time."""
    candidates = [
        candidate
        for candidate in available_snapshot_times(scan_time.date(), snapshot_dir)
        if candidate <= scan_time
    ]
    if not candidates:
        raise FileNotFoundError(f"No snapshot found at or before {scan_time}")
    nearest = candidates[-1]
    if scan_time - nearest > timedelta(minutes=max_age_minutes):
        raise FileNotFoundError(f"Latest snapshot {nearest} is older than {max_age_minutes} minutes")
    return nearest


def github_snapshot_store() -> GitHubSnapshotStore | None:
    config = load_github_snapshot_config()
    if not config:
        return None
    return GitHubSnapshotStore(config)
