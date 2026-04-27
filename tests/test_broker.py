from __future__ import annotations

import pandas as pd
import pytest

from trading_app.broker import (
    ALPACA_PAPER_BASE_URL,
    AlpacaPaperBroker,
    Order,
    OrderRiskLimits,
    OrderSide,
    PaperBroker,
    orders_for_target_weights,
    validate_orders_against_risk_limits,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.requests = []

    def request(self, method, url, timeout, **kwargs):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "timeout": timeout,
                **kwargs,
            }
        )
        if url.endswith("/v2/account"):
            return FakeResponse(
                {
                    "id": "account-id",
                    "status": "ACTIVE",
                    "cash": "1000.50",
                    "portfolio_value": "1200.00",
                    "buying_power": "2000.00",
                }
            )
        if url.endswith("/v2/positions"):
            return FakeResponse(
                [
                    {
                        "symbol": "AAPL",
                        "qty": "2",
                        "market_value": "400.00",
                        "avg_entry_price": "190.00",
                    }
                ]
            )
        return FakeResponse({"id": "order-id", "status": "accepted"})


def test_paper_broker_buy_and_sell_updates_cash_and_positions() -> None:
    broker = PaperBroker(cash=1_000.0)
    prices = pd.Series({"AAA": 10.0})

    broker.submit_order(Order("AAA", OrderSide.BUY, 5), prices)
    broker.submit_order(Order("AAA", OrderSide.SELL, 2), prices)

    assert broker.cash == 970.0
    assert broker.positions == {"AAA": 3}
    assert broker.portfolio_value(prices) == 1_000.0


def test_paper_broker_rejects_oversell() -> None:
    broker = PaperBroker(cash=1_000.0, positions={"AAA": 1})

    with pytest.raises(ValueError, match="Cannot sell"):
        broker.submit_order(Order("AAA", OrderSide.SELL, 2), pd.Series({"AAA": 10.0}))


def test_orders_for_target_weights_generates_integer_orders() -> None:
    orders = orders_for_target_weights(
        cash=1_000.0,
        positions={},
        prices=pd.Series({"AAA": 25.0}),
        target_weights=pd.Series({"AAA": 0.5}),
    )

    assert orders == [Order("AAA", OrderSide.BUY, 4)]


def test_orders_for_target_weights_sells_removed_positions() -> None:
    orders = orders_for_target_weights(
        cash=0.0,
        positions={"AAA": 10},
        prices=pd.Series({"AAA": 10.0}),
        target_weights=pd.Series(dtype=float),
    )

    assert orders == [Order("AAA", OrderSide.SELL, 10)]


def test_alpaca_paper_broker_rejects_live_endpoint() -> None:
    with pytest.raises(ValueError, match="Only Alpaca paper"):
        AlpacaPaperBroker(
            api_key="key",
            secret_key="secret",
            base_url="https://api.alpaca.markets",
        )


def test_alpaca_paper_broker_fetches_account_and_positions() -> None:
    session = FakeSession()
    broker = AlpacaPaperBroker(api_key="key", secret_key="secret", session=session)

    account = broker.fetch_account()
    positions = broker.fetch_positions()

    assert broker.base_url == ALPACA_PAPER_BASE_URL
    assert account.cash == 1000.50
    assert account.status == "ACTIVE"
    assert positions[0].ticker == "AAPL"
    assert positions[0].quantity == 2


def test_alpaca_paper_broker_dry_run_does_not_submit_order() -> None:
    session = FakeSession()
    broker = AlpacaPaperBroker(api_key="key", secret_key="secret", dry_run=True, session=session)

    result = broker.submit_order(Order("AAPL", OrderSide.BUY, 1), approved=True)

    assert result["dry_run"] is True
    assert session.requests == []


def test_alpaca_paper_broker_requires_manual_approval() -> None:
    broker = AlpacaPaperBroker(api_key="key", secret_key="secret", dry_run=True, session=FakeSession())

    with pytest.raises(PermissionError, match="Manual approval"):
        broker.submit_order(Order("AAPL", OrderSide.BUY, 1))


def test_alpaca_paper_broker_submits_only_to_paper_endpoint() -> None:
    session = FakeSession()
    broker = AlpacaPaperBroker(api_key="key", secret_key="secret", dry_run=False, session=session)

    result = broker.submit_order(Order("AAPL", OrderSide.BUY, 1), approved=True)

    assert result["status"] == "accepted"
    assert session.requests[0]["method"] == "POST"
    assert session.requests[0]["url"] == f"{ALPACA_PAPER_BASE_URL}/v2/orders"
    assert session.requests[0]["json"]["symbol"] == "AAPL"
    assert session.requests[0]["json"]["side"] == "buy"


def test_orders_for_target_weights_enforces_position_count_and_weight_caps() -> None:
    orders = orders_for_target_weights(
        cash=10_000.0,
        positions={},
        prices=pd.Series(
            {
                "AAA": 100.0,
                "BBB": 100.0,
                "CCC": 100.0,
                "DDD": 100.0,
                "EEE": 100.0,
                "FFF": 100.0,
            }
        ),
        target_weights=pd.Series(
            {
                "AAA": 0.5,
                "BBB": 0.5,
                "CCC": 0.5,
                "DDD": 0.5,
                "EEE": 0.5,
                "FFF": 0.5,
            }
        ),
    )

    assert len(orders) == 5
    assert all(order.quantity == 10 for order in orders)


def test_order_risk_limits_reject_caps_above_hard_limits() -> None:
    with pytest.raises(ValueError, match="10% hard safety cap"):
        OrderRiskLimits(max_position_weight=0.11).validate()

    with pytest.raises(ValueError, match="5-position hard safety cap"):
        OrderRiskLimits(max_open_positions=6).validate()


def test_validate_orders_rejects_projected_position_over_hard_cap() -> None:
    with pytest.raises(ValueError, match="exceeds 10% hard cap"):
        validate_orders_against_risk_limits(
            cash=10_000.0,
            positions={},
            prices=pd.Series({"AAA": 100.0}),
            orders=[Order("AAA", OrderSide.BUY, 11)],
        )
