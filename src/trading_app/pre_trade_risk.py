from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd

from trading_app.broker import AlpacaOpenOrder, Order, OrderSide
from trading_app.intraday_loop import is_market_open, market_now


MAX_ACCOUNT_RISK_PER_TRADE = 0.01
MAX_OPEN_POSITIONS = 5
MAX_TICKER_ALLOCATION = 0.20
DEFAULT_FRESHNESS_LIMIT = timedelta(minutes=30)


@dataclass(frozen=True)
class PreTradeValidationResult:
    ticker: str
    side: str
    quantity: int
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_pre_trade_orders(
    orders: list[Order],
    *,
    account_equity: float,
    positions: dict[str, int],
    prices: pd.Series,
    price_timestamps: dict[str, datetime],
    open_orders: list[AlpacaOpenOrder],
    auto_trade: bool,
    dry_run: bool,
    manual_confirmed: bool,
    now: datetime | None = None,
    stop_prices: dict[str, float] | None = None,
    freshness_limit: timedelta = DEFAULT_FRESHNESS_LIMIT,
) -> list[PreTradeValidationResult]:
    if account_equity <= 0:
        return [
            _reject(order, "Account equity must be positive")
            for order in orders
        ]

    current_time = market_now(now)
    seen_tickers: set[str] = set()
    projected_positions = positions.copy()
    open_order_tickers = {order.ticker for order in open_orders}
    results: list[PreTradeValidationResult] = []

    for order in orders:
        rejection = _first_rejection_reason(
            order=order,
            current_time=current_time,
            account_equity=account_equity,
            projected_positions=projected_positions,
            prices=prices,
            price_timestamps=price_timestamps,
            open_order_tickers=open_order_tickers,
            seen_tickers=seen_tickers,
            auto_trade=auto_trade,
            dry_run=dry_run,
            manual_confirmed=manual_confirmed,
            stop_prices=stop_prices or {},
            freshness_limit=freshness_limit,
        )
        seen_tickers.add(order.ticker)
        if rejection:
            results.append(_reject(order, rejection))
            continue

        _apply_projected_order(projected_positions, order)
        results.append(
            PreTradeValidationResult(
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                accepted=True,
                reason="Accepted by pre-trade risk validation",
            )
        )

    return results


def accepted_orders(
    orders: list[Order],
    results: list[PreTradeValidationResult],
) -> list[Order]:
    return [
        order
        for order, result in zip(orders, results)
        if result.accepted
    ]


def validation_results_to_dataframe(results: list[PreTradeValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_dict() for result in results])


def _first_rejection_reason(
    *,
    order: Order,
    current_time: datetime,
    account_equity: float,
    projected_positions: dict[str, int],
    prices: pd.Series,
    price_timestamps: dict[str, datetime],
    open_order_tickers: set[str],
    seen_tickers: set[str],
    auto_trade: bool,
    dry_run: bool,
    manual_confirmed: bool,
    stop_prices: dict[str, float],
    freshness_limit: timedelta,
) -> str | None:
    if order.ticker in seen_tickers:
        return "Duplicate order for ticker in the same cycle"
    if not is_market_open(current_time):
        return "Market is closed"
    if order.ticker in open_order_tickers:
        return "Open Alpaca order already exists for this ticker"
    if order.ticker not in prices or float(prices[order.ticker]) <= 0:
        return "Missing valid current price"

    timestamp = price_timestamps.get(order.ticker)
    if timestamp is None:
        return "Missing price timestamp"
    age = current_time - _as_market_time(timestamp)
    if age < timedelta(0) or age > freshness_limit:
        return "Market data is stale"

    if order.side == OrderSide.BUY and projected_positions.get(order.ticker, 0) > 0:
        return "Ticker already has an open position"
    if order.side == OrderSide.BUY and _open_position_count(projected_positions) >= MAX_OPEN_POSITIONS:
        return "Max 5 open positions reached"

    price = float(prices[order.ticker])
    projected_quantity = projected_positions.get(order.ticker, 0)
    if order.side == OrderSide.BUY:
        projected_quantity += order.quantity
    elif order.side == OrderSide.SELL:
        projected_quantity = max(projected_quantity - order.quantity, 0)

    allocation = projected_quantity * price / account_equity
    if allocation > MAX_TICKER_ALLOCATION + 1e-9:
        return "Ticker allocation would exceed 20% of account equity"

    if order.side == OrderSide.BUY:
        stop_price = stop_prices.get(order.ticker)
        risk_per_share = price - stop_price if stop_price is not None else price
        trade_risk = max(risk_per_share, 0.0) * order.quantity
        if trade_risk > account_equity * MAX_ACCOUNT_RISK_PER_TRADE + 1e-9:
            return "Trade risk would exceed 1% of account equity"

    if not dry_run and not auto_trade and not manual_confirmed:
        return "Live mode requires manual confirmation unless AUTO_TRADE=true"

    return None


def _apply_projected_order(projected_positions: dict[str, int], order: Order) -> None:
    current_quantity = projected_positions.get(order.ticker, 0)
    if order.side == OrderSide.BUY:
        projected_positions[order.ticker] = current_quantity + order.quantity
    else:
        remaining = max(current_quantity - order.quantity, 0)
        if remaining:
            projected_positions[order.ticker] = remaining
        else:
            projected_positions.pop(order.ticker, None)


def _open_position_count(positions: dict[str, int]) -> int:
    return sum(1 for quantity in positions.values() if quantity > 0)


def _as_market_time(value: datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.to_pydatetime().astimezone(current_market_timezone())


def current_market_timezone():
    return market_now().tzinfo


def _reject(order: Order, reason: str) -> PreTradeValidationResult:
    return PreTradeValidationResult(
        ticker=order.ticker,
        side=order.side.value,
        quantity=order.quantity,
        accepted=False,
        reason=reason,
    )
