from __future__ import annotations

from typing import Any

import pandas as pd

from trading_app.scanner2.config import Scanner2Config


SCAN_KEYS = ["08_00", "08_30", "08_45", "09_00", "09_15", "09_30", "09_45"]


def build_final_dataframe(
    results_by_time: dict[str, dict[str, dict[str, Any]]],
    previous_day_stats: dict[str, dict[str, Any]],
    config: Scanner2Config,
) -> pd.DataFrame:
    """Build the scanner2 output table from staged scan results."""
    del config
    tickers = sorted({ticker for rows in results_by_time.values() for ticker in rows})
    rows = []
    for ticker in tickers:
        base = previous_day_stats.get(ticker, {})
        row: dict[str, Any] = {
            "ticker": ticker,
            "prev_close_4pm": base.get("prev_close_4pm"),
            "prev_day_volume": base.get("prev_day_volume"),
            "premarket_high": None,
            "current_vs_premarket_high_pct": None,
            "final_rank": None,
            "still_qualified": False,
            "error": "",
        }
        errors = []
        for scan_key in SCAN_KEYS:
            result = results_by_time.get(scan_key, {}).get(ticker, {})
            label = scan_key.replace("_", "_")
            if scan_key == "09_45":
                row["price_9_45"] = result.get("current_price")
                row["volume_9_30_to_9_45"] = result.get("volume_9_30_to_9_45")
                row["final_premarket_volume_4am_to_9_30"] = result.get("premarket_volume")
                row["change_pct_9_45"] = result.get("change_pct")
                row["qualified_9_45"] = bool(result.get("qualified", False))
            else:
                row[f"price_{label}"] = result.get("current_price")
                row[f"premarket_volume_4am_to_{label}"] = result.get("premarket_volume")
                row[f"change_pct_{label}"] = result.get("change_pct")
                row[f"qualified_{label}"] = bool(result.get("qualified", False))
            if result.get("premarket_high") is not None:
                row["premarket_high"] = max(
                    [value for value in [row.get("premarket_high"), result.get("premarket_high")] if value is not None]
                )
            if result.get("error"):
                errors.append(f"{scan_key}: {result['error']}")

        final_price = row.get("price_9_45")
        premarket_high = row.get("premarket_high")
        if final_price is not None and premarket_high:
            row["current_vs_premarket_high_pct"] = (final_price - premarket_high) / premarket_high * 100.0
        row["still_qualified"] = bool(row.get("qualified_9_45"))
        row["error"] = "; ".join(errors)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=output_columns())
    if frame.empty:
        return frame
    qualified = frame[frame["still_qualified"]].sort_values("change_pct_9_45", ascending=False)
    for rank, index in enumerate(qualified.index, start=1):
        frame.loc[index, "final_rank"] = rank
    return frame.sort_values(["still_qualified", "change_pct_9_45"], ascending=[False, False], na_position="last")


def output_columns() -> list[str]:
    return [
        "ticker",
        "prev_close_4pm",
        "prev_day_volume",
        "price_8_00",
        "premarket_volume_4am_to_8_00",
        "change_pct_8_00",
        "qualified_8_00",
        "price_8_30",
        "premarket_volume_4am_to_8_30",
        "change_pct_8_30",
        "qualified_8_30",
        "price_8_45",
        "premarket_volume_4am_to_8_45",
        "change_pct_8_45",
        "qualified_8_45",
        "price_9_00",
        "premarket_volume_4am_to_9_00",
        "change_pct_9_00",
        "qualified_9_00",
        "price_9_15",
        "premarket_volume_4am_to_9_15",
        "change_pct_9_15",
        "qualified_9_15",
        "price_9_30",
        "premarket_volume_4am_to_9_30",
        "change_pct_9_30",
        "qualified_9_30",
        "price_9_45",
        "volume_9_30_to_9_45",
        "final_premarket_volume_4am_to_9_30",
        "change_pct_9_45",
        "qualified_9_45",
        "premarket_high",
        "current_vs_premarket_high_pct",
        "final_rank",
        "still_qualified",
        "error",
    ]
