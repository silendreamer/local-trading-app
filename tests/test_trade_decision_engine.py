from __future__ import annotations

import pandas as pd

from trading_app.broker import OrderRiskLimits
from trading_app.trade_decision_engine import (
    IntradayDecisionConfig,
    IntradayPositionState,
    TradeDecisionAction,
    decide_intraday_trade,
)


def make_ohlc(closes: list[float], volume: float = 1_000.0) -> pd.DataFrame:
    close = pd.Series(closes, index=pd.date_range("2026-04-27 09:30", periods=len(closes), freq="15min"))
    return pd.DataFrame(
        {
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": volume,
        }
    )


def test_decision_engine_buys_when_entry_conditions_are_met() -> None:
    data = make_ohlc([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6])

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions={},
        account_equity=100_000.0,
        risk_limits=OrderRiskLimits(max_position_weight=0.10, max_open_positions=5),
        config=IntradayDecisionConfig(
            momentum_lookback=4,
            trend_window=8,
            min_momentum_return=0.003,
        ),
    )

    assert decision.action == TradeDecisionAction.BUY
    assert "Trend and momentum conditions are met" in decision.reason


def test_decision_engine_holds_when_max_open_positions_reached() -> None:
    data = make_ohlc([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6])
    positions = {"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1, "EEE": 1}

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions=positions,
        account_equity=100_000.0,
        risk_limits=OrderRiskLimits(max_position_weight=0.10, max_open_positions=5),
    )

    assert decision.action == TradeDecisionAction.HOLD
    assert decision.reason == "Max open positions reached"


def test_decision_engine_sells_when_stop_loss_is_hit() -> None:
    data = make_ohlc([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 97.0])

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions={"AAPL": 10},
        account_equity=100_000.0,
        position_state=IntradayPositionState(quantity=10, entry_price=100.0),
    )

    assert decision.action == TradeDecisionAction.SELL
    assert decision.reason == "Stop loss is hit"


def test_decision_engine_does_not_buy_when_ticker_is_already_held() -> None:
    data = make_ohlc([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6])

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions={"AAPL": 10},
        account_equity=100_000.0,
        position_state=IntradayPositionState(quantity=10, entry_price=100.0),
    )

    assert decision.action != TradeDecisionAction.BUY
    assert decision.reason == "No exit condition is met"


def test_decision_engine_sells_when_trailing_stop_is_hit() -> None:
    data = make_ohlc([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0])

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions={"AAPL": 10},
        account_equity=100_000.0,
        position_state=IntradayPositionState(
            quantity=10,
            entry_price=100.0,
            highest_price_since_entry=105.0,
        ),
    )

    assert decision.action == TradeDecisionAction.SELL
    assert decision.reason == "Trailing stop is hit"


def test_decision_engine_sells_on_strategy_exit_signal() -> None:
    data = make_ohlc([105.0, 104.5, 104.0, 103.5, 103.0, 102.5, 102.0, 101.0, 100.0])

    decision = decide_intraday_trade(
        "AAPL",
        data,
        positions={"AAPL": 10},
        account_equity=100_000.0,
        position_state=IntradayPositionState(quantity=10, entry_price=99.0),
    )

    assert decision.action == TradeDecisionAction.SELL
    assert decision.reason == "Strategy exit signal: price fell below moving average"


def test_decision_engine_skips_missing_intraday_data() -> None:
    decision = decide_intraday_trade(
        "AAPL",
        pd.DataFrame(),
        positions={},
        account_equity=100_000.0,
    )

    assert decision.action == TradeDecisionAction.SKIP
    assert decision.reason == "Missing complete intraday OHLC data"
