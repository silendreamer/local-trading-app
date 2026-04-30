from __future__ import annotations

import pandas as pd

from trading_app.broker import Order, OrderSide, PaperBroker
from trading_app.order_execution import (
    OrderExecutionConfig,
    build_execution_order,
    execute_orders,
    limit_price_for,
    risk_based_quantity,
)


class FakeSubmitter:
    def __init__(self):
        self.submitted = []

    def submit_order(self, order: Order, approved: bool = False):
        self.submitted.append((order, approved))
        return {"id": "order-id", "status": "accepted", "symbol": order.ticker}


def test_risk_based_quantity_uses_one_percent_account_risk() -> None:
    quantity = risk_based_quantity(
        account_equity=100_000.0,
        current_price=100.0,
        stop_price=95.0,
    )

    assert quantity == 200


def test_build_buy_execution_order_uses_risk_sizing_and_limit_buffer() -> None:
    order = build_execution_order(
        ticker="AAPL",
        side=OrderSide.BUY,
        current_price=100.0,
        account_equity=100_000.0,
        stop_price=95.0,
        config=OrderExecutionConfig(limit_buffer_pct=0.001),
    )

    assert order.quantity == 200
    assert order.order_type == "limit"
    assert order.limit_price == 100.10


def test_build_sell_execution_order_closes_existing_position() -> None:
    order = build_execution_order(
        ticker="AAPL",
        side=OrderSide.SELL,
        current_price=100.0,
        account_equity=100_000.0,
        existing_quantity=7,
    )

    assert order.quantity == 7
    assert order.limit_price == 99.90


def test_limit_price_for_buy_and_sell() -> None:
    assert limit_price_for(OrderSide.BUY, 100.0, 0.002) == 100.20
    assert limit_price_for(OrderSide.SELL, 100.0, 0.002) == 99.80


def test_execute_orders_dry_run_does_not_submit_and_writes_history(tmp_path) -> None:
    submitter = FakeSubmitter()
    order = Order("AAPL", OrderSide.BUY, 1, limit_price=100.10)

    results = execute_orders(
        submitter,
        [order],
        dry_run=True,
        history_path=tmp_path / "orders.csv",
    )

    assert submitter.submitted == []
    assert results[0].status == "dry_run"
    history = pd.read_csv(tmp_path / "orders.csv")
    assert list(history["event"]) == ["PROPOSED", "SUBMITTED_DRY_RUN"]


def test_execute_orders_submits_when_not_dry_run(tmp_path) -> None:
    submitter = FakeSubmitter()
    order = Order("AAPL", OrderSide.BUY, 1, limit_price=100.10)

    results = execute_orders(
        submitter,
        [order],
        dry_run=False,
        history_path=tmp_path / "orders.csv",
    )

    assert submitter.submitted == [(order, True)]
    assert results[0].status == "accepted"


def test_paper_broker_accepts_order_with_limit_metadata() -> None:
    broker = PaperBroker(cash=1_000.0)
    fill = broker.submit_order(Order("AAA", OrderSide.BUY, 1, limit_price=10.01), pd.Series({"AAA": 10.0}))

    assert fill.ticker == "AAA"
