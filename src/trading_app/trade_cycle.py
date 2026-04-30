from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from trading_app.broker import (
    AlpacaOpenOrder,
    AlpacaPaperBroker,
    AlpacaPosition,
    Order,
    OrderRiskLimits,
    OrderSide,
    dedupe_orders_by_ticker,
)
from trading_app.config import AppSettings, PROJECT_ROOT, load_tickers
from trading_app.data import MarketDataRequest, fetch_ohlc
from trading_app.intraday_loop import is_market_open, market_now
from trading_app.order_execution import (
    OrderExecutionConfig,
    OrderExecutionResult,
    build_execution_order,
    execute_orders,
)
from trading_app.pre_trade_risk import (
    PreTradeValidationResult,
    accepted_orders,
    validate_pre_trade_orders,
)
from trading_app.scanners.price_action import PriceActionScanResult, scan_price_action
from trading_app.trade_decision_engine import (
    IntradayPositionState,
    TradeDecision,
    TradeDecisionAction,
    decide_intraday_trade,
)


@dataclass(frozen=True)
class TradeCycleSummary:
    status: str
    reason: str
    timestamp: str
    market_open: bool
    ticker_count: int
    decision_count: int
    proposed_order_count: int
    accepted_order_count: int
    submitted_order_count: int
    dry_run: bool
    auto_trade: bool
    scan_results: list[dict[str, object]]
    decisions: list[dict[str, object]]
    proposed_orders: list[dict[str, object]]
    risk_results: list[dict[str, object]]
    order_results: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BrokerFactory = Callable[[AppSettings], AlpacaPaperBroker | None]
TickerLoader = Callable[[], list[str]]
MarketDataFetcher = Callable[[MarketDataRequest], dict[str, pd.DataFrame]]


def run_trade_cycle(
    settings: AppSettings,
    *,
    approved: bool = False,
    now: datetime | None = None,
    ticker_loader: TickerLoader = load_tickers,
    market_data_fetcher: MarketDataFetcher = fetch_ohlc,
    broker_factory: BrokerFactory | None = None,
    output_dir: Path | None = None,
) -> TradeCycleSummary:
    current_time = market_now(now)
    if not is_market_open(current_time):
        return _summary(
            status="skipped",
            reason="Market is closed",
            timestamp=current_time,
            market_open=False,
            dry_run=settings.dry_run,
            auto_trade=settings.auto_trade,
        )

    tickers = ticker_loader()
    broker = _build_broker(settings, broker_factory)
    account, positions, open_orders = _fetch_broker_state(settings, broker)
    market_data = _fetch_latest_market_data(tickers, current_time, market_data_fetcher)
    scan_results = scan_price_action(tickers, now=current_time, fetcher=market_data_fetcher)
    decisions = _generate_decisions(tickers, market_data, positions, account.portfolio_value)
    proposed_orders = _create_proposed_orders(decisions, positions, account.portfolio_value)
    risk_results = _validate_orders(
        proposed_orders,
        account_equity=account.portfolio_value,
        positions=positions,
        open_orders=open_orders,
        market_data=market_data,
        settings=settings,
        approved=approved,
        now=current_time,
    )
    accepted = accepted_orders(proposed_orders, risk_results)
    order_results = _execute_cycle_orders(
        broker,
        accepted,
        settings=settings,
        approved=approved,
    )
    _save_cycle_outputs(scan_results, decisions, proposed_orders, risk_results, order_results, output_dir)
    return _summary(
        status="completed",
        reason="Trade cycle completed",
        timestamp=current_time,
        market_open=True,
        dry_run=settings.dry_run,
        auto_trade=settings.auto_trade,
        ticker_count=len(tickers),
        scan_results=[result.to_dict() for result in scan_results],
        decisions=[decision.to_dict() for decision in decisions],
        proposed_orders=[_order_to_dict(order) for order in proposed_orders],
        risk_results=[result.to_dict() for result in risk_results],
        order_results=[result.to_dict() for result in order_results],
    )


def _build_broker(settings: AppSettings, broker_factory: BrokerFactory | None) -> AlpacaPaperBroker | None:
    if broker_factory is not None:
        return broker_factory(settings)
    if not settings.alpaca_api_key or not settings.alpaca_secret_key or not settings.alpaca_paper:
        return None
    return AlpacaPaperBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        dry_run=settings.dry_run,
    )


@dataclass(frozen=True)
class _CycleAccount:
    portfolio_value: float


def _fetch_broker_state(
    settings: AppSettings,
    broker: AlpacaPaperBroker | None,
) -> tuple[_CycleAccount, dict[str, AlpacaPosition], list[AlpacaOpenOrder]]:
    if broker is None:
        return _CycleAccount(settings.paper_starting_cash), {}, []
    account = broker.fetch_account()
    positions = {
        position.ticker: position
        for position in broker.fetch_positions()
        if position.quantity > 0
    }
    return _CycleAccount(account.portfolio_value), positions, broker.fetch_open_orders()


def _fetch_latest_market_data(
    tickers: list[str],
    current_time: datetime,
    fetcher: MarketDataFetcher,
) -> dict[str, pd.DataFrame]:
    return fetcher(
        MarketDataRequest(
            tickers=tickers,
            start=current_time.date() - timedelta(days=5),
            end=current_time.date() + timedelta(days=1),
            interval="15m",
        )
    )


def _generate_decisions(
    tickers: list[str],
    market_data: dict[str, pd.DataFrame],
    positions: dict[str, AlpacaPosition],
    account_equity: float,
) -> list[TradeDecision]:
    position_quantities = {ticker: position.quantity for ticker, position in positions.items()}
    return [
        decide_intraday_trade(
            ticker,
            market_data.get(ticker, pd.DataFrame()),
            positions=position_quantities,
            account_equity=account_equity,
            risk_limits=OrderRiskLimits(max_open_positions=5),
            position_state=_position_state(positions[ticker]) if ticker in positions else None,
        )
        for ticker in tickers
    ]


def _create_proposed_orders(
    decisions: list[TradeDecision],
    positions: dict[str, AlpacaPosition],
    account_equity: float,
) -> list[Order]:
    orders: list[Order] = []
    for decision in decisions:
        if decision.action not in {TradeDecisionAction.BUY, TradeDecisionAction.SELL}:
            continue
        if decision.current_price is None:
            continue
        side = OrderSide.BUY if decision.action == TradeDecisionAction.BUY else OrderSide.SELL
        position = positions.get(decision.ticker)
        stop_price = decision.current_price * 0.95 if side == OrderSide.BUY else None
        try:
            orders.append(
                build_execution_order(
                    ticker=decision.ticker,
                    side=side,
                    current_price=decision.current_price,
                    account_equity=account_equity,
                    existing_quantity=position.quantity if position else 0,
                    stop_price=stop_price,
                    config=OrderExecutionConfig(),
                )
            )
        except ValueError:
            continue
    return dedupe_orders_by_ticker(orders)


def _validate_orders(
    orders: list[Order],
    *,
    account_equity: float,
    positions: dict[str, AlpacaPosition],
    open_orders: list[AlpacaOpenOrder],
    market_data: dict[str, pd.DataFrame],
    settings: AppSettings,
    approved: bool,
    now: datetime,
) -> list[PreTradeValidationResult]:
    prices = pd.Series(
        {
            ticker: float(frame["Close"].dropna().iloc[-1])
            for ticker, frame in market_data.items()
            if not frame.empty and "Close" in frame
        }
    )
    timestamps = {
        ticker: pd.Timestamp(frame.dropna().index[-1]).to_pydatetime()
        for ticker, frame in market_data.items()
        if not frame.empty
    }
    stop_prices = {
        order.ticker: float(prices[order.ticker]) * 0.95
        for order in orders
        if order.side == OrderSide.BUY and order.ticker in prices
    }
    return validate_pre_trade_orders(
        orders,
        account_equity=account_equity,
        positions={ticker: position.quantity for ticker, position in positions.items()},
        prices=prices,
        price_timestamps=timestamps,
        open_orders=open_orders,
        auto_trade=settings.auto_trade,
        dry_run=settings.dry_run,
        manual_confirmed=approved,
        now=now,
        stop_prices=stop_prices,
    )


def _execute_cycle_orders(
    broker: AlpacaPaperBroker | None,
    orders: list[Order],
    *,
    settings: AppSettings,
    approved: bool,
) -> list[OrderExecutionResult]:
    if not orders:
        return []
    if settings.dry_run:
        return execute_orders(_NullBroker(), orders, dry_run=True)
    if broker is None or not settings.alpaca_paper:
        return []
    if not approved and not settings.auto_trade:
        return []
    return execute_orders(broker, orders, dry_run=False)


def _save_cycle_outputs(
    scan_results: list[PriceActionScanResult],
    decisions: list[TradeDecision],
    proposed_orders: list[Order],
    risk_results: list[PreTradeValidationResult],
    order_results: list[OrderExecutionResult],
    output_dir: Path | None,
) -> None:
    target = output_dir or PROJECT_ROOT / "logs"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.to_dict() for result in scan_results]).to_csv(target / "latest_scan_results.csv", index=False)
    pd.DataFrame([decision.to_dict() for decision in decisions]).to_csv(target / "latest_trade_decisions.csv", index=False)
    pd.DataFrame([_order_to_dict(order) for order in proposed_orders]).to_csv(target / "latest_proposed_orders.csv", index=False)
    pd.DataFrame([result.to_dict() for result in risk_results]).to_csv(target / "latest_risk_results.csv", index=False)
    pd.DataFrame([result.to_dict() for result in order_results]).to_csv(target / "latest_order_results.csv", index=False)


def _summary(
    *,
    status: str,
    reason: str,
    timestamp: datetime,
    market_open: bool,
    dry_run: bool,
    auto_trade: bool,
    ticker_count: int = 0,
    scan_results: list[dict[str, object]] | None = None,
    decisions: list[dict[str, object]] | None = None,
    proposed_orders: list[dict[str, object]] | None = None,
    risk_results: list[dict[str, object]] | None = None,
    order_results: list[dict[str, object]] | None = None,
) -> TradeCycleSummary:
    scan_results = scan_results or []
    decisions = decisions or []
    proposed_orders = proposed_orders or []
    risk_results = risk_results or []
    order_results = order_results or []
    return TradeCycleSummary(
        status=status,
        reason=reason,
        timestamp=timestamp.isoformat(),
        market_open=market_open,
        ticker_count=ticker_count,
        decision_count=len(decisions),
        proposed_order_count=len(proposed_orders),
        accepted_order_count=sum(1 for result in risk_results if result.get("accepted")),
        submitted_order_count=len(order_results),
        dry_run=dry_run,
        auto_trade=auto_trade,
        scan_results=scan_results,
        decisions=decisions,
        proposed_orders=proposed_orders,
        risk_results=risk_results,
        order_results=order_results,
    )


def _position_state(position: AlpacaPosition) -> IntradayPositionState:
    current_price = position.current_price or position.average_entry_price
    return IntradayPositionState(
        quantity=position.quantity,
        entry_price=position.average_entry_price,
        highest_price_since_entry=max(position.average_entry_price, current_price),
    )


def _order_to_dict(order: Order) -> dict[str, object]:
    return {
        "ticker": order.ticker,
        "side": order.side.value,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "limit_price": order.limit_price,
    }


class _NullBroker:
    def submit_order(self, order: Order, approved: bool = False) -> dict[str, object]:
        return {"dry_run": True, "symbol": order.ticker, "status": "not_submitted"}
