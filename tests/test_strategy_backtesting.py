from __future__ import annotations

import pandas as pd

from trading_app.backtesting import run_backtest
from trading_app.risk import RiskLimits, apply_risk_limits
from trading_app.strategies.strategy import MovingAverageConfig, moving_average_signals, target_equal_weights


def test_moving_average_strategy_produces_weights() -> None:
    prices = pd.DataFrame(
        {"AAA": [10, 11, 12, 13, 14, 15], "BBB": [15, 14, 13, 12, 11, 10]},
        index=pd.date_range("2024-01-01", periods=6),
    )

    signals = moving_average_signals(prices, MovingAverageConfig(short_window=2, long_window=3))
    weights = target_equal_weights(signals)

    assert weights.loc["2024-01-06", "AAA"] == 1.0
    assert weights.loc["2024-01-06", "BBB"] == 0.0


def test_backtest_uses_prior_day_weights() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 121.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    weights = pd.DataFrame(
        {"AAA": [1.0, 1.0, 1.0]},
        index=prices.index,
    )

    result = run_backtest(prices, weights, initial_cash=100.0)

    assert result.equity_curve.iloc[0] == 100.0
    assert round(result.equity_curve.iloc[-1], 2) == 121.0


def test_risk_limits_cap_position_weights() -> None:
    weights = pd.DataFrame(
        {"AAA": [0.8], "BBB": [0.2]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    capped = apply_risk_limits(weights, RiskLimits(max_position_weight=0.25, max_gross_exposure=0.5))

    assert capped.iloc[0].max() <= 0.25
    assert round(float(capped.iloc[0].sum()), 2) == 0.45
