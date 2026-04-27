from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MomentumTradingConfig:
    lookback_window: int = 60
    trend_window: int = 120
    min_momentum_return: float = 0.05
    max_positions: int = 5

    def validate(self) -> None:
        if self.lookback_window < 2:
            raise ValueError("lookback_window must be at least 2")
        if self.trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if self.min_momentum_return < 0:
            raise ValueError("min_momentum_return cannot be negative")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")


def momentum_scores(prices: pd.DataFrame, config: MomentumTradingConfig) -> pd.DataFrame:
    """Measure recent price momentum as lookback-window percentage return."""
    config.validate()
    if prices.empty:
        raise ValueError("Price data is empty")
    return prices.pct_change(config.lookback_window)


def momentum_trading_signals(
    prices: pd.DataFrame,
    config: MomentumTradingConfig,
) -> pd.DataFrame:
    """Generate long-only signals for confirmed positive momentum.

    The strategy buys the strongest recent performers that are also above a
    longer moving-average trend filter, then exits when momentum or trend weakens.
    """
    config.validate()
    if prices.empty:
        raise ValueError("Price data is empty")

    scores = momentum_scores(prices, config)
    trend = prices.rolling(config.trend_window, min_periods=config.trend_window).mean()
    eligible = (scores >= config.min_momentum_return) & (prices > trend)
    ranks = scores.where(eligible).rank(axis=1, method="first", ascending=False)
    signals = (ranks <= config.max_positions).astype(int)
    return signals.where(scores.notna() & trend.notna(), 0)
