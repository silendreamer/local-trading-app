from __future__ import annotations

from dataclasses import dataclass
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


MARKET_TIMEZONE_NAME = "America/New_York"
MARKET_TIMEZONE = ZoneInfo(MARKET_TIMEZONE_NAME)
SCAN_TIMES = ["08:00", "08:30", "08:45", "09:00", "09:15", "09:30", "09:45"]

MIN_PRICE = 2.0
MAX_PRICE = 50.0
MIN_GAP_PCT = 20.0
MIN_PREV_DAY_VOLUME = 500_000
MIN_PREMARKET_VOLUME = 100_000
PREMARKET_VOLUME_TO_PREV_DAY_RATIO = 0.50
TOP_N = 50
REQUEST_SLEEP_SECONDS = 0.25


@dataclass(frozen=True)
class Scanner2Config:
    polygon_api_key: str
    min_price: float = MIN_PRICE
    max_price: float = MAX_PRICE
    min_gap_pct: float = MIN_GAP_PCT
    min_prev_day_volume: int = MIN_PREV_DAY_VOLUME
    min_premarket_volume: int = MIN_PREMARKET_VOLUME
    premarket_volume_to_prev_day_ratio: float = PREMARKET_VOLUME_TO_PREV_DAY_RATIO
    top_n: int = TOP_N
    request_sleep_seconds: float = REQUEST_SLEEP_SECONDS
    scan_times: tuple[str, ...] = tuple(SCAN_TIMES)
    timezone_name: str = MARKET_TIMEZONE_NAME


def load_config() -> Scanner2Config:
    """Load scanner2 settings from environment variables."""
    load_dotenv()
    return Scanner2Config(
        polygon_api_key=os.getenv("POLYGON_API_KEY", ""),
        min_price=float(os.getenv("SCANNER2_MIN_PRICE", str(MIN_PRICE))),
        max_price=float(os.getenv("SCANNER2_MAX_PRICE", str(MAX_PRICE))),
        min_gap_pct=float(os.getenv("SCANNER2_MIN_GAP_PCT", str(MIN_GAP_PCT))),
        min_prev_day_volume=int(os.getenv("SCANNER2_MIN_PREV_DAY_VOLUME", str(MIN_PREV_DAY_VOLUME))),
        min_premarket_volume=int(os.getenv("SCANNER2_MIN_PREMARKET_VOLUME", str(MIN_PREMARKET_VOLUME))),
        premarket_volume_to_prev_day_ratio=float(
            os.getenv(
                "SCANNER2_PREMARKET_VOLUME_TO_PREV_DAY_RATIO",
                str(PREMARKET_VOLUME_TO_PREV_DAY_RATIO),
            )
        ),
        top_n=int(os.getenv("SCANNER2_TOP_N", str(TOP_N))),
        request_sleep_seconds=float(os.getenv("SCANNER2_REQUEST_SLEEP_SECONDS", str(REQUEST_SLEEP_SECONDS))),
    )
