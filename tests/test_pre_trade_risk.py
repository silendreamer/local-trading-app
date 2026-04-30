from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from trading_app.broker import AlpacaOpenOrder, Order, OrderSide
from trading_app.intraday_loop import MARKET_TIMEZONE
from trading_app.pre_trade_risk import accepted_orders, validate_pre_trade_orders


MARKET_TIME = datetime(2026, 4, 27, 10, 0, tzinfo=MARKET_TIMEZONE)


def validate(order: Order, **overrides):
    kwargs = {
        "account_equity": 100_000.0,
        "positions": {},
        "prices": pd.Series({order.ticker: 100.0}),
        "price_timestamps": {order.ticker: MARKET_TIME},
        "open_orders": [],
        "auto_trade": False,
        "dry_run": True,
        "manual_confirmed": False,
        "now": MARKET_TIME,
    }
    kwargs.update(overrides)
    return validate_pre_trade_orders([order], **kwargs)[0]


def test_accepts_dry_run_order_with_fresh_data_and_market_open() -> None:
    result = validate(Order("AAPL", OrderSide.BUY, 5))

    assert result.accepted
    assert result.reason == "Accepted by pre-trade risk validation"


def test_rejects_trade_risk_above_one_percent() -> None:
    result = validate(Order("AAPL", OrderSide.BUY, 11))

    assert not result.accepted
    assert result.reason == "Trade risk would exceed 1% of account equity"


def test_rejects_more_than_five_open_positions() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        positions={"AAA": 1, "BBB": 1, "CCC": 1, "DDD": 1, "EEE": 1},
    )

    assert not result.accepted
    assert result.reason == "Max 5 open positions reached"


def test_rejects_allocation_above_twenty_percent() -> None:
    result = validate(Order("AAPL", OrderSide.BUY, 201))

    assert not result.accepted
    assert result.reason == "Ticker allocation would exceed 20% of account equity"


def test_rejects_stale_data() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        price_timestamps={"AAPL": MARKET_TIME - timedelta(minutes=31)},
    )

    assert not result.accepted
    assert result.reason == "Market data is stale"


def test_rejects_outside_market_hours() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        now=datetime(2026, 4, 27, 16, 1, tzinfo=MARKET_TIMEZONE),
    )

    assert not result.accepted
    assert result.reason == "Market is closed"


def test_rejects_buy_for_existing_position() -> None:
    result = validate(Order("AAPL", OrderSide.BUY, 1), positions={"AAPL": 1})

    assert not result.accepted
    assert result.reason == "Ticker already has an open position"


def test_rejects_open_alpaca_order_for_same_ticker() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        open_orders=[AlpacaOpenOrder("1", "AAPL", OrderSide.BUY, 1, "new")],
    )

    assert not result.accepted
    assert result.reason == "Open Alpaca order already exists for this ticker"


def test_rejects_live_mode_without_manual_confirmation_or_auto_trade() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        dry_run=False,
        auto_trade=False,
        manual_confirmed=False,
    )

    assert not result.accepted
    assert result.reason == "Live mode requires manual confirmation unless AUTO_TRADE=true"


def test_accepts_live_mode_with_manual_confirmation() -> None:
    result = validate(
        Order("AAPL", OrderSide.BUY, 1),
        dry_run=False,
        auto_trade=False,
        manual_confirmed=True,
    )

    assert result.accepted


def test_rejects_duplicate_order_in_same_cycle() -> None:
    orders = [
        Order("AAPL", OrderSide.BUY, 1),
        Order("AAPL", OrderSide.BUY, 1),
    ]
    results = validate_pre_trade_orders(
        orders,
        account_equity=100_000.0,
        positions={},
        prices=pd.Series({"AAPL": 100.0}),
        price_timestamps={"AAPL": MARKET_TIME},
        open_orders=[],
        auto_trade=False,
        dry_run=True,
        manual_confirmed=False,
        now=MARKET_TIME,
    )

    assert results[0].accepted
    assert not results[1].accepted
    assert results[1].reason == "Duplicate order for ticker in the same cycle"
    assert accepted_orders(orders, results) == [orders[0]]
