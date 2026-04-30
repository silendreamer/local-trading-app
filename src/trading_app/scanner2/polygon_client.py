from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import time
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
POLYGON_BASE_URL = "https://api.polygon.io"


@dataclass(frozen=True)
class PolygonResult:
    ok: bool
    data: Any = None
    error: str = ""
    status_code: int | None = None


class PolygonRestClient:
    """Small Polygon REST client with retry handling for scanner2."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        request_sleep_seconds: float = 0.25,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY is required")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.request_sleep_seconds = request_sleep_seconds
        self.max_retries = max_retries

    def get_all_tickers_snapshot(self) -> PolygonResult:
        """Fetch all US stock ticker snapshots."""
        return self._get("/v2/snapshot/locale/us/markets/stocks/tickers", {"include_otc": "false"})

    def get_grouped_daily_bars(self, target_date: str) -> PolygonResult:
        """Fetch grouped daily bars for a market date formatted as YYYY-MM-DD."""
        return self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{target_date}", {"adjusted": "true"})

    def get_intraday_minute_bars(self, ticker: str, from_datetime: datetime, to_datetime: datetime) -> PolygonResult:
        """Fetch 1-minute aggregate bars for a ticker between timezone-aware datetimes."""
        from_ms = int(from_datetime.timestamp() * 1000)
        to_ms = int(to_datetime.timestamp() * 1000)
        return self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{from_ms}/{to_ms}",
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": "50000",
            },
        )

    def get_latest_price(self, ticker: str) -> PolygonResult:
        """Fetch latest trade price for a ticker."""
        return self._get(f"/v2/last/trade/{ticker}", {})

    def _get(self, path: str, params: dict[str, Any]) -> PolygonResult:
        url = f"{POLYGON_BASE_URL}{path}"
        request_params = {**params, "apiKey": self.api_key}
        for attempt in range(1, self.max_retries + 1):
            if self.request_sleep_seconds > 0:
                time.sleep(self.request_sleep_seconds)
            try:
                response = self.session.get(url, params=request_params, timeout=30)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    wait_seconds = min(2 ** attempt, 10)
                    LOGGER.warning("Polygon request retry path=%s status=%s wait=%s", path, response.status_code, wait_seconds)
                    time.sleep(wait_seconds)
                    continue
                if response.status_code == 403:
                    return PolygonResult(
                        ok=False,
                        error=f"Polygon 403 Forbidden for {path}. Your plan may not include this endpoint.",
                        status_code=response.status_code,
                    )
                response.raise_for_status()
                payload = response.json()
                status = payload.get("status") if isinstance(payload, dict) else None
                if status in {"ERROR", "NOT_AUTHORIZED"}:
                    return PolygonResult(ok=False, data=payload, error=str(payload), status_code=response.status_code)
                return PolygonResult(ok=True, data=payload, status_code=response.status_code)
            except Exception as exc:
                if attempt >= self.max_retries:
                    LOGGER.exception("Polygon request failed path=%s", path)
                    return PolygonResult(ok=False, error=str(exc))
        return PolygonResult(ok=False, error=f"Polygon request failed for {path}")
