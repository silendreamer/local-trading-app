from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MovingAverageConfig:
    short_window: int = 20
    long_window: int = 50

    def validate(self) -> None:
        if self.short_window < 1:
            raise ValueError("short_window must be at least 1")
        if self.long_window <= self.short_window:
            raise ValueError("long_window must be greater than short_window")


def moving_average_signals(
    prices: pd.DataFrame,
    config: MovingAverageConfig,
) -> pd.DataFrame:
    """Generate long-only signals from a moving-average crossover."""
    config.validate()
    if prices.empty:
        raise ValueError("Price data is empty")

    short_ma = prices.rolling(config.short_window, min_periods=config.short_window).mean()
    long_ma = prices.rolling(config.long_window, min_periods=config.long_window).mean()
    signals = (short_ma > long_ma).astype(int)
    return signals.where(long_ma.notna(), 0)


def target_equal_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """Convert binary long signals to equal-weight target allocations."""
    active_count = signals.sum(axis=1).astype(float)
    active_count = active_count.where(active_count != 0)
    weights = signals.div(active_count, axis=0).fillna(0.0)
    return weights.astype(float)
