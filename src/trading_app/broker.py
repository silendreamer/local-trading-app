from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from trading_app.config import PROJECT_ROOT


ORDER_LOGGER_NAME = "trading_app.orders"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_MAX_POSITION_WEIGHT = 0.10
DEFAULT_MAX_OPEN_POSITIONS = 5


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Order:
    ticker: str
    side: OrderSide
    quantity: int


@dataclass(frozen=True)
class OrderRiskLimits:
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS

    def validate(self) -> None:
        if not 0 < self.max_position_weight <= DEFAULT_MAX_POSITION_WEIGHT:
            raise ValueError("max_position_weight cannot exceed the 10% hard safety cap")
        if not 1 <= self.max_open_positions <= DEFAULT_MAX_OPEN_POSITIONS:
            raise ValueError("max_open_positions cannot exceed the 5-position hard safety cap")


@dataclass(frozen=True)
class AlpacaAccount:
    account_id: str
    status: str
    cash: float
    portfolio_value: float
    buying_power: float


@dataclass(frozen=True)
class AlpacaPosition:
    ticker: str
    quantity: int
    market_value: float
    average_entry_price: float


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime


@dataclass
class PaperBroker:
    """Simple in-memory broker for paper trading only."""

    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def submit_order(self, order: Order, market_prices: pd.Series) -> Fill:
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if order.ticker not in market_prices:
            raise ValueError(f"Missing market price for {order.ticker}")

        price = float(market_prices[order.ticker])
        notional = price * order.quantity
        current_position = self.positions.get(order.ticker, 0)

        if order.side == OrderSide.BUY:
            if notional > self.cash:
                raise ValueError("Insufficient paper cash")
            self.cash -= notional
            self.positions[order.ticker] = current_position + order.quantity
        elif order.side == OrderSide.SELL:
            if order.quantity > current_position:
                raise ValueError("Cannot sell more shares than paper position")
            self.cash += notional
            self.positions[order.ticker] = current_position - order.quantity
        else:
            raise ValueError(f"Unsupported order side: {order.side}")

        if self.positions.get(order.ticker) == 0:
            self.positions.pop(order.ticker, None)

        fill = Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=price,
            timestamp=datetime.now(timezone.utc),
        )
        self.fills.append(fill)
        return fill

    def portfolio_value(self, market_prices: pd.Series) -> float:
        holdings_value = sum(
            quantity * float(market_prices.get(ticker, 0.0))
            for ticker, quantity in self.positions.items()
        )
        return float(self.cash + holdings_value)


class AlpacaPaperBroker:
    """Minimal Alpaca paper-trading client.

    This client is hard-coded to Alpaca's paper endpoint and refuses any other
    base URL. It does not support live-trading endpoints.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        dry_run: bool = True,
        session: requests.Session | None = None,
        base_url: str = ALPACA_PAPER_BASE_URL,
    ) -> None:
        if base_url.rstrip("/") != ALPACA_PAPER_BASE_URL:
            raise ValueError("Only Alpaca paper trading endpoint is allowed")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

        self.base_url = ALPACA_PAPER_BASE_URL
        self.dry_run = dry_run
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Content-Type": "application/json",
            }
        )

    def fetch_account(self) -> AlpacaAccount:
        payload = self._request("GET", "/v2/account")
        return AlpacaAccount(
            account_id=str(payload.get("id", "")),
            status=str(payload.get("status", "")),
            cash=float(payload.get("cash", 0.0)),
            portfolio_value=float(payload.get("portfolio_value", 0.0)),
            buying_power=float(payload.get("buying_power", 0.0)),
        )

    def fetch_positions(self) -> list[AlpacaPosition]:
        payload = self._request("GET", "/v2/positions")
        return [
            AlpacaPosition(
                ticker=str(position.get("symbol", "")),
                quantity=int(float(position.get("qty", 0))),
                market_value=float(position.get("market_value", 0.0)),
                average_entry_price=float(position.get("avg_entry_price", 0.0)),
            )
            for position in payload
        ]

    def submit_order(self, order: Order, approved: bool = False) -> dict[str, Any]:
        if not approved:
            raise PermissionError("Manual approval is required before submitting paper orders")
        if self.dry_run:
            log_order_event("SUBMITTED_DRY_RUN", order)
            return {
                "dry_run": True,
                "symbol": order.ticker,
                "side": order.side.value,
                "qty": order.quantity,
                "status": "not_submitted",
            }

        payload = {
            "symbol": order.ticker,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": "market",
            "time_in_force": "day",
        }
        response = self._request("POST", "/v2/orders", json=payload)
        log_order_event("SUBMITTED", order)
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method=method,
            url=f"{self.base_url}{path}",
            timeout=20,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()


def configure_order_logging(log_path: Path | None = None) -> None:
    logger = logging.getLogger(ORDER_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return

    target = log_path or PROJECT_ROOT / "logs" / "orders.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def log_order_event(event: str, order: Order) -> None:
    configure_order_logging()
    logging.getLogger(ORDER_LOGGER_NAME).info(
        "%s ticker=%s side=%s quantity=%s",
        event,
        order.ticker,
        order.side.value,
        order.quantity,
    )


def log_proposed_orders(orders: list[Order]) -> None:
    for order in orders:
        log_order_event("PROPOSED", order)


def orders_for_target_weights(
    cash: float,
    positions: dict[str, int],
    prices: pd.Series,
    target_weights: pd.Series,
    risk_limits: OrderRiskLimits | None = None,
) -> list[Order]:
    """Create integer-share orders to move toward target weights."""
    if cash < 0:
        raise ValueError("cash cannot be negative")
    limits = risk_limits or OrderRiskLimits()
    limits.validate()

    current_value = cash + sum(
        quantity * float(prices.get(ticker, 0.0)) for ticker, quantity in positions.items()
    )
    orders: list[Order] = []
    safe_target_weights = cap_target_weights(target_weights, limits)

    tradable_tickers = sorted(set(positions) | set(safe_target_weights.index))
    for ticker in tradable_tickers:
        if ticker not in prices or prices[ticker] <= 0:
            continue
        target_weight = float(safe_target_weights.get(ticker, 0.0))
        target_value = current_value * float(target_weight)
        current_position = positions.get(ticker, 0)
        current_position_value = current_position * float(prices[ticker])
        delta_value = target_value - current_position_value
        quantity = int(abs(delta_value) // float(prices[ticker]))
        if quantity == 0:
            continue
        side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
        orders.append(Order(ticker=ticker, side=side, quantity=quantity))

    return orders


def cap_target_weights(target_weights: pd.Series, limits: OrderRiskLimits) -> pd.Series:
    """Apply hard order safety caps to target weights."""
    limits.validate()
    if target_weights.empty:
        return target_weights.copy()
    capped = target_weights.clip(lower=0.0, upper=limits.max_position_weight)
    capped = capped[capped > 0].sort_values(ascending=False).head(limits.max_open_positions)
    return capped


def validate_orders_against_risk_limits(
    cash: float,
    positions: dict[str, int],
    prices: pd.Series,
    orders: list[Order],
    risk_limits: OrderRiskLimits | None = None,
) -> None:
    """Validate projected positions before any paper submission."""
    limits = risk_limits or OrderRiskLimits()
    limits.validate()
    projected_cash = cash
    projected_positions = positions.copy()

    for order in orders:
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if order.ticker not in prices or prices[order.ticker] <= 0:
            raise ValueError(f"Missing valid market price for {order.ticker}")
        price = float(prices[order.ticker])
        notional = price * order.quantity
        current_quantity = projected_positions.get(order.ticker, 0)
        if order.side == OrderSide.BUY:
            projected_cash -= notional
            projected_positions[order.ticker] = current_quantity + order.quantity
        elif order.side == OrderSide.SELL:
            if order.quantity > current_quantity:
                raise ValueError(f"Cannot sell more {order.ticker} than projected position")
            projected_cash += notional
            projected_positions[order.ticker] = current_quantity - order.quantity
        else:
            raise ValueError(f"Unsupported order side: {order.side}")
        if projected_positions.get(order.ticker) == 0:
            projected_positions.pop(order.ticker, None)

    if projected_cash < -0.01:
        raise ValueError("Projected cash would be negative")

    active_positions = {
        ticker: quantity
        for ticker, quantity in projected_positions.items()
        if quantity > 0 and ticker in prices
    }
    if len(active_positions) > limits.max_open_positions:
        raise ValueError("Projected portfolio exceeds 5 open positions")

    projected_equity = projected_cash + sum(
        quantity * float(prices[ticker])
        for ticker, quantity in active_positions.items()
    )
    if projected_equity <= 0:
        raise ValueError("Projected equity must be positive")

    for ticker, quantity in active_positions.items():
        weight = quantity * float(prices[ticker]) / projected_equity
        if weight > limits.max_position_weight + 1e-9:
            raise ValueError(f"Projected {ticker} position exceeds 10% hard cap")
