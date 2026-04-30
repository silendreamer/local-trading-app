from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Any, Protocol

from trading_app.broker import Order, OrderSide, log_order_event
from trading_app.config import PROJECT_ROOT


ORDER_HISTORY_COLUMNS = [
    "timestamp",
    "event",
    "ticker",
    "side",
    "quantity",
    "order_type",
    "limit_price",
    "status",
    "response",
]


class OrderSubmitter(Protocol):
    def submit_order(self, order: Order, approved: bool = False) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OrderExecutionConfig:
    risk_per_trade: float = 0.01
    limit_buffer_pct: float = 0.001

    def validate(self) -> None:
        if not 0 < self.risk_per_trade <= 0.01:
            raise ValueError("risk_per_trade must be in (0, 0.01]")
        if self.limit_buffer_pct < 0:
            raise ValueError("limit_buffer_pct cannot be negative")


@dataclass(frozen=True)
class OrderExecutionResult:
    ticker: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    status: str
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_execution_order(
    *,
    ticker: str,
    side: OrderSide,
    current_price: float,
    account_equity: float,
    existing_quantity: int = 0,
    stop_price: float | None = None,
    configured_quantity: int | None = None,
    config: OrderExecutionConfig | None = None,
) -> Order:
    cfg = config or OrderExecutionConfig()
    cfg.validate()
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")

    if side == OrderSide.BUY:
        quantity = risk_based_quantity(
            account_equity=account_equity,
            current_price=current_price,
            stop_price=stop_price,
            risk_per_trade=cfg.risk_per_trade,
        )
    else:
        quantity = configured_quantity if configured_quantity is not None else existing_quantity

    if quantity < 1:
        raise ValueError("Calculated order quantity must be at least 1")

    return Order(
        ticker=ticker,
        side=side,
        quantity=quantity,
        limit_price=limit_price_for(side, current_price, cfg.limit_buffer_pct),
        order_type="limit",
    )


def risk_based_quantity(
    *,
    account_equity: float,
    current_price: float,
    stop_price: float | None,
    risk_per_trade: float = 0.01,
) -> int:
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if not 0 < risk_per_trade <= 0.01:
        raise ValueError("risk_per_trade must be in (0, 0.01]")

    risk_budget = account_equity * risk_per_trade
    risk_per_share = current_price - stop_price if stop_price is not None else current_price
    if risk_per_share <= 0:
        raise ValueError("stop_price must be below current_price for BUY risk sizing")
    return floor(risk_budget / risk_per_share)


def limit_price_for(side: OrderSide, current_price: float, buffer_pct: float) -> float:
    if buffer_pct < 0:
        raise ValueError("buffer_pct cannot be negative")
    if side == OrderSide.BUY:
        return round(current_price * (1 + buffer_pct), 2)
    return round(current_price * (1 - buffer_pct), 2)


def execute_orders(
    broker: OrderSubmitter,
    orders: list[Order],
    *,
    dry_run: bool,
    history_path: Path | None = None,
) -> list[OrderExecutionResult]:
    results: list[OrderExecutionResult] = []
    for order in orders:
        log_order_event("PROPOSED", order)
        write_order_history("PROPOSED", order, {"status": "proposed"}, history_path)
        if dry_run:
            response = {
                "dry_run": True,
                "symbol": order.ticker,
                "side": order.side.value,
                "qty": order.quantity,
                "type": order.order_type,
                "limit_price": order.limit_price,
                "status": "not_submitted",
            }
            status = "dry_run"
        else:
            response = broker.submit_order(order, approved=True)
            status = str(response.get("status", "submitted"))

        log_order_event("SUBMITTED_DRY_RUN" if dry_run else "SUBMITTED", order)
        write_order_history("SUBMITTED_DRY_RUN" if dry_run else "SUBMITTED", order, response, history_path)
        results.append(
            OrderExecutionResult(
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price,
                status=status,
                response=response,
            )
        )
    return results


def write_order_history(
    event: str,
    order: Order,
    response: dict[str, Any],
    history_path: Path | None = None,
) -> None:
    target = history_path or PROJECT_ROOT / "logs" / "orders.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    file_exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ORDER_HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "status": response.get("status", ""),
                "response": response,
            }
        )
