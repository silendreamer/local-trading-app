from __future__ import annotations

import trading_app.scanner2.background_snapshot as background_snapshot
from trading_app.scanner2.config import Scanner2Config


class FakeThread:
    created = 0

    def __init__(self, **kwargs):
        FakeThread.created += 1
        self.started = False
        self.kwargs = kwargs

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started


def reset_background_state() -> None:
    background_snapshot._THREAD = None
    background_snapshot._STARTED_AT = None
    background_snapshot._MESSAGE = "Snapshot service has not started."
    FakeThread.created = 0


def test_background_snapshot_service_stays_off_when_disabled() -> None:
    reset_background_state()

    status = background_snapshot.start_background_snapshot_service(
        Scanner2Config(polygon_api_key="key"),
        enabled=False,
    )

    assert status.enabled is False
    assert status.running is False
    assert "disabled" in status.message


def test_background_snapshot_service_requires_polygon_key() -> None:
    reset_background_state()

    status = background_snapshot.start_background_snapshot_service(
        Scanner2Config(polygon_api_key=""),
        enabled=True,
    )

    assert status.enabled is True
    assert status.running is False
    assert "POLYGON_API_KEY" in status.message


def test_background_snapshot_service_starts_only_once(monkeypatch) -> None:
    reset_background_state()
    monkeypatch.setattr(background_snapshot.threading, "Thread", FakeThread)

    first = background_snapshot.start_background_snapshot_service(
        Scanner2Config(polygon_api_key="key"),
        enabled=True,
    )
    second = background_snapshot.start_background_snapshot_service(
        Scanner2Config(polygon_api_key="key"),
        enabled=True,
    )

    assert first.running is True
    assert second.running is True
    assert FakeThread.created == 1
