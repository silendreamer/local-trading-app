from __future__ import annotations

import pandas as pd
import pytest

from trading_app.strategies.momentum_strategy import MomentumTradingConfig, momentum_trading_signals


def test_momentum_strategy_selects_strongest_confirmed_momentum() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 104.0, 106.0, 112.0, 120.0],
            "BBB": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "CCC": [120.0, 118.0, 116.0, 114.0, 112.0, 110.0],
        },
        index=pd.date_range("2024-01-01", periods=6),
    )
    config = MomentumTradingConfig(
        lookback_window=3,
        trend_window=4,
        min_momentum_return=0.05,
        max_positions=1,
    )

    signals = momentum_trading_signals(prices, config)

    assert signals.loc["2024-01-06", "AAA"] == 1
    assert signals.loc["2024-01-06", "BBB"] == 0
    assert signals.loc["2024-01-06", "CCC"] == 0


def test_momentum_strategy_waits_for_enough_history() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 105.0, 110.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    config = MomentumTradingConfig(lookback_window=3, trend_window=4)

    signals = momentum_trading_signals(prices, config)

    assert signals.sum().sum() == 0


def test_momentum_strategy_rejects_invalid_config() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 105.0, 110.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    config = MomentumTradingConfig(lookback_window=1)

    with pytest.raises(ValueError, match="lookback_window"):
        momentum_trading_signals(prices, config)
