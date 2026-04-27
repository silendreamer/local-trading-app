from __future__ import annotations

import pandas as pd

from trading_app.strategies.swing_strategy import (
    PositionState,
    SwingStrategyConfig,
    TradeAction,
    generate_swing_signal,
)


def make_ohlc(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, index=pd.date_range("2024-01-01", periods=len(closes)))
    return pd.DataFrame(
        {
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        }
    )


def test_swing_strategy_generates_buy_signal_with_position_size() -> None:
    data = make_ohlc([100, 101, 102, 103, 102, 103, 103, 103])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
    )

    signal = generate_swing_signal(
        data,
        account_equity=10_000.0,
        open_positions=0,
        config=config,
    )

    assert signal.action == TradeAction.BUY
    assert signal.reason == "Price above 200-day MA, 50-day MA rising, RSI between 40 and 65"
    assert signal.rsi == 50.0
    assert signal.atr == 2.0
    assert signal.stop_loss == 99.0
    assert signal.risk_amount == 100.0
    assert signal.shares == 25


def test_swing_strategy_holds_when_max_positions_reached() -> None:
    data = make_ohlc([100, 101, 102, 103, 102, 103, 103, 103])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
        max_open_positions=5,
    )

    signal = generate_swing_signal(
        data,
        account_equity=10_000.0,
        open_positions=5,
        config=config,
    )

    assert signal.action == TradeAction.HOLD
    assert signal.reason == "Max open positions reached"
    assert signal.shares == 0


def test_swing_strategy_holds_when_price_below_trend_filter() -> None:
    data = make_ohlc([110, 109, 108, 107, 106, 105, 104, 103])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
    )

    signal = generate_swing_signal(
        data,
        account_equity=10_000.0,
        open_positions=0,
        config=config,
    )

    assert signal.action == TradeAction.HOLD
    assert signal.reason == "Price is not above 200-day moving average"


def test_swing_strategy_sells_when_stop_loss_hit() -> None:
    data = make_ohlc([100, 101, 102, 103, 102, 103, 103, 103])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
    )

    signal = generate_swing_signal(
        data,
        account_equity=10_000.0,
        open_positions=1,
        position=PositionState(quantity=10, entry_price=105.0, stop_loss=104.0),
        config=config,
    )

    assert signal.action == TradeAction.SELL
    assert signal.reason == "Price hit 2 ATR stop loss"
    assert signal.stop_loss == 104.0


def test_swing_strategy_holds_until_enough_history_exists() -> None:
    data = make_ohlc([100, 101, 102])

    signal = generate_swing_signal(
        data,
        account_equity=10_000.0,
        open_positions=0,
        config=SwingStrategyConfig(),
    )

    assert signal.action == TradeAction.HOLD
    assert signal.reason == "Not enough price history for swing strategy indicators"
