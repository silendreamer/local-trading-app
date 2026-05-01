from __future__ import annotations

import sys
from base64 import b64encode
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_app.backtesting import CostAssumptions, run_backtest, run_swing_strategy_backtest
from trading_app.broker import (
    AlpacaPaperBroker,
    AlpacaPosition,
    Order,
    OrderRiskLimits,
    OrderSide,
    PaperBroker,
    cap_target_weights,
    dedupe_orders_by_ticker,
    log_proposed_orders,
    orders_for_target_weights,
    validate_orders_against_risk_limits,
)
from trading_app.config import load_environment, load_tickers
from trading_app.data import MarketDataRequest, fetch_latest_quote, fetch_ohlc, latest_prices
from trading_app.intraday_loop import (
    TradingLoopManager,
    format_market_time,
    is_market_open,
    market_now,
    next_market_open,
)
from trading_app.logging_config import configure_logging
from trading_app.order_execution import OrderExecutionConfig, build_execution_order, execute_orders
from trading_app.scanners.polygon import (
    PolygonSnapshotScannerConfig,
    polygon_scanner_columns,
    scan_polygon_snapshot,
)
from trading_app.scanners.price_action import results_to_dataframe, scan_price_action
from trading_app.pre_trade_risk import (
    accepted_orders,
    validate_pre_trade_orders,
    validation_results_to_dataframe,
)
from trading_app.risk import RiskLimits, apply_risk_limits
from trading_app.scanner2.background_snapshot import (
    BackgroundSnapshotStatus,
    start_background_snapshot_service,
)
from trading_app.scanner2.config import load_config as load_scanner2_config
from trading_app.scanner2.output_builder import output_columns as scanner2_output_columns
from trading_app.scanner2.polygon_client import PolygonRestClient as Scanner2PolygonRestClient
from trading_app.scanner2.scanner import run_full_scan as run_scanner2_full_scan
from trading_app.scanner2.snapshot_store import capture_snapshot, snapshot_path
from trading_app.strategies.momentum_strategy import MomentumTradingConfig, momentum_trading_signals
from trading_app.strategies.strategy import MovingAverageConfig, moving_average_signals, target_equal_weights
from trading_app.strategies.swing_strategy import SwingStrategyConfig
from trading_app.trade_decision_engine import IntradayPositionState, decide_intraday_trade
from trading_app.trade_cycle import run_trade_cycle


st.set_page_config(page_title="Paper Trading Research", layout="wide")
HARD_ORDER_RISK_LIMITS = OrderRiskLimits()
HEADER_IMAGE = ROOT / "assets" / "trading-header.png"
STRATEGY_MOVING_AVERAGE = "Moving average crossover"
STRATEGY_MOMENTUM = "Momentum trading"
STRATEGY_SWING = "Swing strategy"
STRATEGIES = [STRATEGY_MOVING_AVERAGE, STRATEGY_MOMENTUM, STRATEGY_SWING]
TRADING_LOOP_KEY = "trading_loop_manager"
SCANNER_RESULTS_KEY = "price_action_scanner_results"
TRADE_DECISIONS_KEY = "intraday_trade_decisions"
SUBMITTED_ORDERS_KEY = "submitted_orders"
TRADE_CYCLE_SUMMARY_KEY = "trade_cycle_summary"
SCANNER2_SNAPSHOT_INTERVAL_MINUTES = 15
SCANNER2_SNAPSHOT_POLL_SECONDS = 30
SCANNER2_SNAPSHOT_RETENTION_DAYS = 3


@st.cache_data(ttl=900)
def cached_fetch_ohlc(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    return fetch_ohlc(MarketDataRequest(tickers=tickers, start=start, end=end))


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def ohlc_to_close_prices(ohlc_by_ticker: dict[str, pd.DataFrame], tickers: list[str]) -> pd.DataFrame:
    return pd.concat(
        [ohlc_by_ticker[ticker]["Close"].rename(ticker) for ticker in tickers if ticker in ohlc_by_ticker],
        axis=1,
    ).sort_index()


def render_header() -> None:
    encoded_header = b64encode(HEADER_IMAGE.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="top-banner">
            <img src="data:image/png;base64,{encoded_header}" alt="" />
        </div>
        <style>
            .top-banner {{
                width: 100%;
                height: 100px;
                max-height: 100px;
                overflow: hidden;
            }}
            .top-banner img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Paper Trading Research")
    st.caption("Local Streamlit dashboard for research, backtesting, and Alpaca paper trading only.")


def render_home(settings) -> None:
    st.subheader("Home")
    strategy_col, _ = st.columns([1, 3])
    with strategy_col:
        selected_strategy = st.selectbox(
            "Strategy to research",
            STRATEGIES,
            key="selected_strategy",
        )
    st.write(
        "Pick one strategy at a time, then use the Trade tab to run research with the current controls."
    )
    st.caption(f"Selected strategy: {selected_strategy}")
    cols = st.columns(4)
    cols[0].metric("Mode", "Paper")
    cols[1].metric("DRY_RUN", str(settings.dry_run))
    cols[2].metric("Max position", "10%")
    cols[3].metric("Max open positions", "5")
    st.info("This app is for research and education only. It is not financial advice.")


def render_test_tab() -> None:
    st.subheader("TEST")
    with st.form("test_ticker_price_form"):
        ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not ticker:
            st.warning("Enter a ticker.")
        else:
            try:
                quote = fetch_latest_quote(ticker)
                st.session_state["test_ticker_price_result"] = {
                    "ticker": quote.ticker,
                    "price": quote.price,
                    "timestamp": quote.latest_trading_day,
                    "previous_close": quote.previous_close,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                }
            except Exception as exc:
                st.session_state["test_ticker_price_result"] = {
                    "ticker": ticker,
                    "error": str(exc),
                }

    result = st.session_state.get("test_ticker_price_result")
    if result:
        st.subheader("Ticker Price")
        if "error" in result:
            st.error(f"{result['ticker']}: {result['error']}")
        else:
            st.metric(result["ticker"], f"${result['price']:,.2f}")
            st.caption(f"Latest trading day: {result['timestamp']}")
            details = {
                "previous_close": result.get("previous_close"),
                "change": result.get("change"),
                "change_percent": result.get("change_percent"),
            }
            st.dataframe(pd.DataFrame([details]), use_container_width=True)
    else:
        st.info("Enter a ticker and submit to fetch the latest available price.")


def render_place_order_tab(settings) -> None:
    st.subheader("Place Order")
    st.caption("Manual Alpaca paper limit order entry. Keep dry run enabled to log without submitting.")

    if not settings.alpaca_paper:
        st.error("ALPACA_PAPER=false is not allowed. This app only supports Alpaca paper trading.")
        return
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        st.warning("Add Alpaca paper API credentials to .env before placing orders.")
        return
    with st.form("manual_alpaca_order_form"):
        ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
        quantity_text = st.text_input("Quantity", placeholder="1").strip()
        limit_price_text = st.text_input("Limit price", placeholder="100.00").strip()
        side_value = st.selectbox("Side", [OrderSide.BUY.value, OrderSide.SELL.value])
        order_dry_run = st.checkbox("Dry run", value=True)
        submitted = st.form_submit_button("Submit Order", type="primary")

    if submitted:
        try:
            if not ticker:
                raise ValueError("Ticker is required")
            quantity = int(quantity_text)
            if quantity <= 0:
                raise ValueError("Quantity must be a positive whole number")
            limit_price = float(limit_price_text)
            if limit_price <= 0:
                raise ValueError("Limit price must be positive")

            broker = AlpacaPaperBroker(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                dry_run=order_dry_run,
            )
            open_orders = broker.fetch_open_orders()
            if any(order.ticker == ticker for order in open_orders):
                raise ValueError(f"An open Alpaca order already exists for {ticker}")

            order = Order(
                ticker=ticker,
                side=OrderSide(side_value),
                quantity=quantity,
                limit_price=limit_price,
                order_type="limit",
            )
            response = broker.submit_order(order, approved=True)
            st.session_state["manual_order_result"] = {
                "ticker": ticker,
                "side": order.side.value,
                "quantity": quantity,
                "limit_price": limit_price,
                "dry_run": order_dry_run,
                "response": response,
            }
        except Exception as exc:
            st.session_state["manual_order_result"] = {
                "ticker": ticker,
                "error": str(exc),
            }

    result = st.session_state.get("manual_order_result")
    if result:
        st.subheader("Order Result")
        if "error" in result:
            st.error(f"{result.get('ticker') or 'Order'}: {result['error']}")
        else:
            st.success("Dry-run order logged." if result.get("dry_run") else "Order submitted to Alpaca paper trading.")
            st.dataframe(pd.DataFrame([result]), use_container_width=True)


def render_scanner_tab(settings) -> None:
    st.subheader("Scanner")
    st.caption("Uses Polygon's full-market stock snapshot, then filters by gap, volume, and price.")
    control_cols = st.columns(5)
    gap_threshold = control_cols[0].number_input("Min gap %", min_value=0.0, value=5.0, step=0.5)
    min_premarket_volume = control_cols[1].number_input(
        "Min volume",
        min_value=0,
        value=100_000,
        step=10_000,
    )
    min_price = control_cols[2].number_input("Min price", min_value=0.01, value=2.0, step=0.5)
    max_price = control_cols[3].number_input("Max price", min_value=0.01, value=20.0, step=0.5)
    top_n = control_cols[4].number_input("Top N", min_value=1, max_value=100, value=50, step=1)

    results = st.session_state.get(
        "scanner_table_results",
        pd.DataFrame(columns=polygon_scanner_columns()),
    )
    st.dataframe(results, use_container_width=True)

    if st.button("SCAN", type="primary", use_container_width=True):
        try:
            if not settings.polygon_api_key:
                st.warning("Add POLYGON_API_KEY to .env before running the Polygon snapshot scanner.")
                return
            with st.spinner("Fetching Polygon full-market snapshot and filtering top movers..."):
                st.session_state["scanner_table_results"] = scan_polygon_snapshot(
                    settings.polygon_api_key,
                    config=PolygonSnapshotScannerConfig(
                        min_gap_pct=float(gap_threshold),
                        min_volume=int(min_premarket_volume),
                        min_price=float(min_price),
                        max_price=float(max_price),
                        top_n=int(top_n),
                    ),
                )
            st.session_state["scanner_last_run"] = format_market_time(market_now())
            st.rerun()
        except Exception as exc:
            st.session_state["scanner_table_results"] = pd.DataFrame(
                [{"ticker": "", "error": str(exc)}],
                columns=polygon_scanner_columns(),
            )
            st.warning(str(exc))

    last_run = st.session_state.get("scanner_last_run")
    if last_run:
        st.caption(f"Last scan: {last_run}")

def render_scanner2_tab(settings) -> None:
    st.subheader("Scanner2")
    st.caption("Polygon.io premarket momentum watchlist generator only. This tab does not place trades.")
    render_scanner2_snapshot_service_status()
    control_cols = st.columns(7)
    min_price = control_cols[0].number_input("Min price", min_value=0.01, value=2.0, step=0.5, key="scanner2_min_price")
    max_price = control_cols[1].number_input("Max price", min_value=0.01, value=50.0, step=1.0, key="scanner2_max_price")
    min_gap = control_cols[2].number_input("Min gap %", min_value=0.0, value=20.0, step=1.0, key="scanner2_min_gap")
    min_prev_volume = control_cols[3].number_input(
        "Min prev volume",
        min_value=0,
        value=500_000,
        step=50_000,
        key="scanner2_min_prev_volume",
    )
    min_premarket_volume = control_cols[4].number_input(
        "Min premarket volume",
        min_value=0,
        value=100_000,
        step=10_000,
        key="scanner2_min_premarket_volume",
    )
    volume_ratio = control_cols[5].number_input(
        "PM/prev volume ratio",
        min_value=0.0,
        value=0.50,
        step=0.05,
        key="scanner2_volume_ratio",
    )
    top_n = control_cols[6].number_input("Top N", min_value=1, max_value=200, value=50, step=5, key="scanner2_top_n")

    results = st.session_state.get("scanner2_results", pd.DataFrame(columns=scanner2_output_columns()))
    st.dataframe(results, use_container_width=True)

    capture_col, run_col = st.columns(2)
    with capture_col:
        if st.button("CAPTURE SNAPSHOT NOW", use_container_width=True):
            if not settings.polygon_api_key:
                st.warning("Add POLYGON_API_KEY to .env before capturing snapshots.")
                return
            try:
                config = replace(
                    load_scanner2_config(),
                    polygon_api_key=settings.polygon_api_key,
                )
                client = Scanner2PolygonRestClient(
                    config.polygon_api_key,
                    request_sleep_seconds=config.request_sleep_seconds,
                )
                path = capture_snapshot(client, scan_time=market_now(), overwrite=True)
                st.success(f"Saved snapshot: {path}")
            except Exception as exc:
                st.warning(str(exc))

    with run_col:
        if st.button("RUN SCANNER2", type="primary", use_container_width=True):
            if not settings.polygon_api_key:
                st.warning("Add POLYGON_API_KEY to .env before running Scanner2.")
                return
            try:
                config = replace(
                    load_scanner2_config(),
                    polygon_api_key=settings.polygon_api_key,
                    min_price=float(min_price),
                    max_price=float(max_price),
                    min_gap_pct=float(min_gap),
                    min_prev_day_volume=int(min_prev_volume),
                    min_premarket_volume=int(min_premarket_volume),
                    premarket_volume_to_prev_day_ratio=float(volume_ratio),
                    top_n=int(top_n),
                )
                with st.spinner("Analyzing persisted Polygon snapshots..."):
                    st.session_state["scanner2_results"] = run_scanner2_full_scan(config=config)
                st.session_state["scanner2_last_run"] = format_market_time(market_now())
                st.rerun()
            except Exception as exc:
                st.session_state["scanner2_results"] = pd.DataFrame(
                    [{"ticker": "", "error": str(exc)}],
                    columns=scanner2_output_columns(),
                )
                st.warning(str(exc))

    last_run = st.session_state.get("scanner2_last_run")
    if last_run:
        st.caption(f"Last Scanner2 run: {last_run}")
    st.caption(f"Current snapshot path: {snapshot_path(market_now())}")


def scanner2_auto_snapshot_enabled(settings) -> bool:
    """Only auto-run the snapshot worker outside local development."""
    return settings.app_env.strip().lower() != "local"


def start_scanner2_snapshot_worker(settings) -> BackgroundSnapshotStatus:
    config = replace(
        load_scanner2_config(),
        polygon_api_key=settings.polygon_api_key,
    )
    return start_background_snapshot_service(
        config,
        enabled=scanner2_auto_snapshot_enabled(settings),
        interval_minutes=SCANNER2_SNAPSHOT_INTERVAL_MINUTES,
        poll_seconds=SCANNER2_SNAPSHOT_POLL_SECONDS,
        retention_days=SCANNER2_SNAPSHOT_RETENTION_DAYS,
    )


def render_scanner2_snapshot_service_status() -> None:
    status = st.session_state.get("scanner2_snapshot_service_status")
    if not isinstance(status, BackgroundSnapshotStatus):
        return

    status_cols = st.columns(4)
    status_cols[0].metric("Auto snapshots", "On" if status.enabled else "Off")
    status_cols[1].metric("Service", "Running" if status.running else "Stopped")
    status_cols[2].metric("Interval", f"{status.interval_minutes} min")
    status_cols[3].metric("Retention", f"{SCANNER2_SNAPSHOT_RETENTION_DAYS} days")
    st.caption(f"Started: {format_market_time(status.started_at)}")
    if status.message:
        if status.running:
            st.success(status.message)
        else:
            st.info(status.message)


def get_trading_loop_manager() -> TradingLoopManager:
    if TRADING_LOOP_KEY not in st.session_state:
        st.session_state[TRADING_LOOP_KEY] = TradingLoopManager()
    return st.session_state[TRADING_LOOP_KEY]


def check_configured_tickers(tickers: list[str]) -> int:
    settings = load_environment()
    summary = run_trade_cycle(settings)
    return int(summary.decision_count)


def position_state_from_alpaca(position: AlpacaPosition) -> IntradayPositionState:
    highest_price = max(position.average_entry_price, position.current_price or position.average_entry_price)
    return IntradayPositionState(
        quantity=position.quantity,
        entry_price=position.average_entry_price,
        highest_price_since_entry=highest_price,
    )


def run_trade_decision_engine(
    tickers: list[str],
    account_equity: float,
    alpaca_positions: list[AlpacaPosition],
) -> pd.DataFrame:
    current_date = market_now().date()
    ohlc_by_ticker = fetch_ohlc(
        MarketDataRequest(
            tickers=tickers,
            start=current_date - timedelta(days=5),
            end=current_date + timedelta(days=1),
            interval="15m",
        )
    )
    position_by_ticker = {
        position.ticker: position
        for position in alpaca_positions
        if position.quantity > 0
    }
    position_quantities = {
        ticker: position.quantity
        for ticker, position in position_by_ticker.items()
    }
    decisions = []
    decided_tickers: set[str] = set()
    for ticker in tickers:
        if ticker in decided_tickers:
            continue
        alpaca_position = position_by_ticker.get(ticker)
        decision = decide_intraday_trade(
            ticker,
            ohlc_by_ticker.get(ticker, pd.DataFrame()),
            positions=position_quantities,
            account_equity=account_equity,
            risk_limits=HARD_ORDER_RISK_LIMITS,
            position_state=position_state_from_alpaca(alpaca_position) if alpaca_position else None,
        ).to_dict()
        if alpaca_position:
            decision.update(
                {
                    "quantity": alpaca_position.quantity,
                    "average_entry_price": alpaca_position.average_entry_price,
                    "market_value": alpaca_position.market_value,
                    "unrealized_pl": alpaca_position.unrealized_pl,
                    "alpaca_current_price": alpaca_position.current_price,
                    "position_mode": "exit",
                }
            )
        else:
            decision.update(
                {
                    "quantity": 0,
                    "average_entry_price": None,
                    "market_value": None,
                    "unrealized_pl": None,
                    "alpaca_current_price": None,
                    "position_mode": "entry",
                }
            )
        decisions.append(decision)
        decided_tickers.add(ticker)
    return pd.DataFrame(decisions)


def fetch_current_alpaca_positions(settings) -> tuple[list[AlpacaPosition], str]:
    if not settings.alpaca_paper:
        return [], "ALPACA_PAPER=false is not allowed; decision engine is using no broker positions."
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        return [], "Alpaca credentials are not configured; decision engine is using no broker positions."
    broker = AlpacaPaperBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        dry_run=settings.dry_run,
    )
    return broker.fetch_positions(), "Fetched current Alpaca paper positions before generating decisions."


def countdown_to(value) -> str:
    if value is None:
        return "-"
    remaining = value - market_now()
    total_seconds = max(int(remaining.total_seconds()), 0)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_order_history(limit: int = 50) -> pd.DataFrame:
    path = ROOT / "logs" / "orders.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).tail(limit)


def alpaca_positions_to_dataframe(positions: list[AlpacaPosition]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": position.ticker,
                "quantity": position.quantity,
                "market_value": position.market_value,
                "average_entry_price": position.average_entry_price,
                "unrealized_pl": position.unrealized_pl,
                "current_price": position.current_price,
            }
            for position in positions
        ]
    )


def render_trade_tab(settings, tickers: list[str]) -> None:
    manager = get_trading_loop_manager()
    current_time = market_now()
    market_open = is_market_open(current_time)
    snapshot = manager.snapshot()

    st.subheader("Trading Status")
    if not settings.dry_run or settings.auto_trade:
        st.warning(
            "Trading safety warning: "
            f"DRY_RUN={settings.dry_run}, AUTO_TRADE={settings.auto_trade}. "
            "Review every order before allowing submissions."
        )
    st.caption(
        "PAPER trading is the default. The loop scans during US market hours and order submission remains gated by risk checks."
    )

    status_cols = st.columns(5)
    status_cols[0].metric("Trading status", snapshot.status)
    status_cols[1].metric("Current market status", "open" if market_open else "closed")
    status_cols[2].metric("Last scan time", format_market_time(snapshot.last_check_time))
    status_cols[3].metric("Next scheduled scan time", format_market_time(snapshot.next_check_time))
    status_cols[4].metric("Countdown", countdown_to(snapshot.next_check_time))

    st.write(
        pd.DataFrame(
            [
                {
                    "trading_status": snapshot.status,
                    "current_market_status": "open" if market_open else "closed",
                    "last_check_time": format_market_time(snapshot.last_check_time),
                    "next_check_time": format_market_time(snapshot.next_check_time),
                    "countdown_to_next_scan": countdown_to(snapshot.next_check_time),
                    "checked_tickers": snapshot.checked_ticker_count,
                    "current_market_time": format_market_time(current_time),
                    "DRY_RUN": settings.dry_run,
                    "AUTO_TRADE": settings.auto_trade,
                    "ALPACA_PAPER": settings.alpaca_paper,
                }
            ]
        )
    )

    if not market_open:
        st.warning(
            "Trading loop cannot start outside US market hours. "
            f"Next market open: {format_market_time(next_market_open(current_time))}."
        )
    if snapshot.last_message:
        st.info(snapshot.last_message)
    if snapshot.last_error:
        st.warning(snapshot.last_error)

    start_col, stop_col = st.columns([1, 1])
    with start_col:
        if st.button(
            "Start Trading",
            type="primary",
            disabled=snapshot.running or not market_open,
            use_container_width=True,
        ):
            message = manager.start(
                tickers,
                dry_run=settings.dry_run,
                alpaca_paper=settings.alpaca_paper,
                check_tickers=check_configured_tickers,
            )
            st.success(message)
            st.rerun()

    with stop_col:
        if st.button(
            "Stop Trading",
            disabled=not snapshot.running,
            use_container_width=True,
        ):
            st.warning(manager.stop())
            st.rerun()

    st.subheader("Live Price Action Scanner")
    st.caption(
        "Fetches recent 15-minute intraday candles for configured tickers, validates freshness, "
        "and returns signal-only scanner output. This scanner does not place trades."
    )
    if st.button("Run Scanner Now", use_container_width=True):
        st.session_state[SCANNER_RESULTS_KEY] = scan_price_action(tickers)
        st.rerun()

    scanner_results = st.session_state.get(SCANNER_RESULTS_KEY, [])
    st.subheader("Latest Signals")
    if scanner_results:
        st.dataframe(results_to_dataframe(scanner_results), use_container_width=True)
    else:
        st.info("No scanner results yet. Run the scanner to fetch latest intraday price action.")

    st.subheader("Intraday Trade Decision Engine")
    st.caption(
        "Evaluates BUY, SELL, HOLD, or SKIP decisions from intraday candles. "
        "This panel returns decisions only and does not submit orders."
    )
    if st.button("Run Decision Engine Now", use_container_width=True):
        if not is_market_open(market_now()):
            st.session_state[TRADE_DECISIONS_KEY] = pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "action": "SKIP",
                        "reason": "Market is closed",
                    }
                    for ticker in tickers
                ]
            )
        else:
            try:
                current_positions, position_message = fetch_current_alpaca_positions(settings)
                st.session_state[TRADE_DECISIONS_KEY] = run_trade_decision_engine(
                    tickers,
                    account_equity=float(settings.paper_starting_cash),
                    alpaca_positions=current_positions,
                )
                st.session_state["trade_decision_position_message"] = position_message
            except Exception as exc:
                st.session_state[TRADE_DECISIONS_KEY] = pd.DataFrame()
                st.session_state["trade_decision_position_message"] = f"Unable to run decision engine: {exc}"
        st.rerun()

    if st.button("Run Full Trade Cycle Now", use_container_width=True):
        summary = run_trade_cycle(settings, approved=True)
        st.session_state[TRADE_CYCLE_SUMMARY_KEY] = summary.to_dict()
        st.session_state[TRADE_DECISIONS_KEY] = pd.DataFrame(summary.decisions)
        st.session_state[SUBMITTED_ORDERS_KEY] = pd.DataFrame(summary.order_results)
        st.rerun()

    cycle_summary = st.session_state.get(TRADE_CYCLE_SUMMARY_KEY)
    if cycle_summary:
        st.subheader("Latest Trade Cycle Summary")
        st.dataframe(pd.DataFrame([cycle_summary]), use_container_width=True)

    decision_results = st.session_state.get(TRADE_DECISIONS_KEY)
    decision_position_message = st.session_state.get("trade_decision_position_message")
    if decision_position_message:
        st.info(decision_position_message)
    if isinstance(decision_results, pd.DataFrame) and not decision_results.empty:
        st.dataframe(decision_results, use_container_width=True)
    else:
        st.info("No trade decisions yet. Run the decision engine to evaluate configured tickers.")

    st.subheader("Current Alpaca Positions")
    try:
        current_positions, position_message = fetch_current_alpaca_positions(settings)
        st.caption(position_message)
        if current_positions:
            st.dataframe(alpaca_positions_to_dataframe(current_positions), use_container_width=True)
        else:
            st.info("No current Alpaca positions available.")
    except Exception as exc:
        st.warning(f"Unable to fetch Alpaca positions: {exc}")

    st.subheader("Proposed Trades")
    if isinstance(decision_results, pd.DataFrame) and not decision_results.empty and "action" in decision_results:
        proposed_trades = decision_results[decision_results["action"].isin(["BUY", "SELL"])]
        if proposed_trades.empty:
            st.info("No BUY or SELL decisions proposed.")
        else:
            st.dataframe(proposed_trades, use_container_width=True)
    else:
        st.info("No proposed trades yet.")

    st.subheader("Submitted Orders")
    submitted_orders = st.session_state.get(SUBMITTED_ORDERS_KEY)
    if isinstance(submitted_orders, pd.DataFrame) and not submitted_orders.empty:
        st.dataframe(submitted_orders, use_container_width=True)
    else:
        history = load_order_history()
        if history.empty:
            st.info("No submitted order history found.")
        else:
            st.dataframe(history, use_container_width=True)

    st.subheader("Configured Tickers")
    st.dataframe(pd.DataFrame({"ticker": tickers}), use_container_width=True)


def main() -> None:
    settings = load_environment()
    configure_logging(settings.log_level)
    st.session_state["scanner2_snapshot_service_status"] = start_scanner2_snapshot_worker(settings)
    tickers = load_tickers()

    render_header()
    home_tab, trade_tab, test_tab, place_order_tab, scanner_tab, scanner2_tab = st.tabs(
        ["HOME", "TRADE", "TEST", "PLACE ORDER", "SCANNER", "SCANNER2"]
    )
    with home_tab:
        render_home(settings)

    with trade_tab:
        render_trade_tab(settings, tickers)

    with test_tab:
        render_test_tab()

    with place_order_tab:
        render_place_order_tab(settings)

    with scanner_tab:
        render_scanner_tab(settings)

    with scanner2_tab:
        render_scanner2_tab(settings)

    with home_tab:
        selected_strategy = st.session_state.get("selected_strategy", STRATEGY_MOVING_AVERAGE)
        with st.sidebar:
            st.header("Research Controls")
            st.caption(f"Researching: {selected_strategy}")
            selected_tickers = st.multiselect(
                "Tickers",
                tickers,
                default=tickers,
                max_selections=20,
            )
            end_date = st.date_input("End date", value=date.today())
            start_date = st.date_input("Start date", value=end_date - timedelta(days=365 * 3))
            short_window = st.slider("Short moving average", 5, 100, 20, step=5)
            long_window = st.slider("Long moving average", 20, 250, 50, step=5)
            momentum_lookback = st.slider("Momentum lookback days", 20, 180, 60, step=10)
            momentum_trend = st.slider("Momentum trend days", 50, 250, 120, step=10)
            momentum_min_return = st.slider("Minimum momentum return", 0.0, 0.5, 0.05, step=0.01)
            max_position = st.slider(
                "Max position weight",
                0.01,
                HARD_ORDER_RISK_LIMITS.max_position_weight,
                HARD_ORDER_RISK_LIMITS.max_position_weight,
                step=0.01,
            )
            st.caption("Hard safety caps: 10% per position, 5 open positions.")
            slippage_bps = st.number_input("Slippage bps", min_value=0.0, value=5.0, step=1.0)
            transaction_cost_bps = st.number_input("Transaction cost bps", min_value=0.0, value=2.0, step=1.0)
            starting_cash = st.number_input(
                "Paper starting cash",
                min_value=1_000.0,
                value=float(settings.paper_starting_cash),
                step=1_000.0,
            )
            run_research = st.button("Run research", type="primary", use_container_width=True)
            if st.button("Reset research", use_container_width=True):
                st.session_state["research_ready"] = False
    
        if run_research:
            st.session_state["research_ready"] = True
    
        if not selected_tickers:
            st.warning("Select at least one ticker.")
            return
        if isinstance(start_date, date) and isinstance(end_date, date) and start_date >= end_date:
            st.warning("Start date must be before end date.")
            return
        if selected_strategy == STRATEGY_MOVING_AVERAGE and long_window <= short_window:
            st.warning("Long moving average must be greater than short moving average.")
            return
        if selected_strategy == STRATEGY_MOMENTUM and momentum_trend <= momentum_lookback:
            st.warning("Momentum trend days must be greater than momentum lookback days.")
            return
        if not st.session_state.get("research_ready", False):
            st.info("Choose research controls, then run the paper-trading analysis.")
            return
    
        try:
            alpaca_client = None
            alpaca_account = None
            alpaca_positions = []
            alpaca_position_map: dict[str, int] = {}
            if not settings.alpaca_paper:
                st.warning("ALPACA_PAPER=false is not allowed. Alpaca integration is disabled.")
            elif settings.alpaca_api_key and settings.alpaca_secret_key:
                try:
                    alpaca_client = AlpacaPaperBroker(
                        api_key=settings.alpaca_api_key,
                        secret_key=settings.alpaca_secret_key,
                        dry_run=settings.dry_run,
                    )
                    alpaca_account = alpaca_client.fetch_account()
                    alpaca_positions = alpaca_client.fetch_positions()
                    alpaca_position_map = {
                        position.ticker: position.quantity
                        for position in alpaca_positions
                        if position.quantity != 0
                    }
                except Exception as exc:
                    st.warning(f"Alpaca paper account unavailable: {exc}")
                    alpaca_client = None
    
            requested_tickers = sorted(set(selected_tickers) | {"SPY"})
            ohlc_by_ticker = cached_fetch_ohlc(requested_tickers, start_date, end_date)
            missing_tickers = sorted(set(selected_tickers) - set(ohlc_by_ticker))
            if missing_tickers:
                st.warning(f"No OHLC data returned for: {', '.join(missing_tickers)}")
            if "SPY" not in ohlc_by_ticker:
                raise ValueError("SPY benchmark data was not returned")
    
            strategy_ohlc = {
                ticker: ohlc_by_ticker[ticker]
                for ticker in selected_tickers
                if ticker in ohlc_by_ticker
            }
            prices = ohlc_to_close_prices(strategy_ohlc, selected_tickers)
            safe_max_position = min(float(max_position), HARD_ORDER_RISK_LIMITS.max_position_weight)
            current_prices = latest_prices(prices)
            broker_cash = alpaca_account.cash if alpaca_account else starting_cash
            broker_positions = alpaca_position_map if alpaca_account else {}
            broker = PaperBroker(cash=broker_cash, positions=broker_positions.copy())

            selected_result = None
            swing_result = None
            latest_target_weights = pd.Series(dtype="float64")
            if selected_strategy == STRATEGY_MOVING_AVERAGE:
                signals = moving_average_signals(
                    prices,
                    MovingAverageConfig(short_window=short_window, long_window=long_window),
                )
                raw_weights = target_equal_weights(signals)
                selected_weights = apply_risk_limits(
                    raw_weights,
                    RiskLimits(max_position_weight=safe_max_position, max_gross_exposure=1.0),
                )
                selected_result = run_backtest(prices, selected_weights, initial_cash=starting_cash)
                latest_target_weights = cap_target_weights(
                    selected_weights.iloc[-1].reindex(current_prices.index).fillna(0.0),
                    HARD_ORDER_RISK_LIMITS,
                )
            elif selected_strategy == STRATEGY_MOMENTUM:
                signals = momentum_trading_signals(
                    prices,
                    MomentumTradingConfig(
                        lookback_window=int(momentum_lookback),
                        trend_window=int(momentum_trend),
                        min_momentum_return=float(momentum_min_return),
                        max_positions=HARD_ORDER_RISK_LIMITS.max_open_positions,
                    ),
                )
                raw_weights = target_equal_weights(signals)
                selected_weights = apply_risk_limits(
                    raw_weights,
                    RiskLimits(max_position_weight=safe_max_position, max_gross_exposure=1.0),
                )
                selected_result = run_backtest(prices, selected_weights, initial_cash=starting_cash)
                latest_target_weights = cap_target_weights(
                    selected_weights.iloc[-1].reindex(current_prices.index).fillna(0.0),
                    HARD_ORDER_RISK_LIMITS,
                )
            else:
                swing_result = run_swing_strategy_backtest(
                    strategy_ohlc,
                    ohlc_by_ticker["SPY"],
                    initial_cash=starting_cash,
                    config=SwingStrategyConfig(),
                    costs=CostAssumptions(
                        slippage_bps=float(slippage_bps),
                        transaction_cost_bps=float(transaction_cost_bps),
                    ),
                )

            if selected_strategy in {STRATEGY_MOVING_AVERAGE, STRATEGY_MOMENTUM}:
                proposed_orders = dedupe_orders_by_ticker(orders_for_target_weights(
                    cash=broker.cash,
                    positions=broker.positions,
                    prices=current_prices,
                    target_weights=latest_target_weights,
                    risk_limits=HARD_ORDER_RISK_LIMITS,
                ))
                validate_orders_against_risk_limits(
                    cash=broker.cash,
                    positions=broker.positions,
                    prices=current_prices,
                    orders=proposed_orders,
                    risk_limits=HARD_ORDER_RISK_LIMITS,
                )
                log_proposed_orders(proposed_orders)
            else:
                proposed_orders = []
        except Exception as exc:
            st.error(f"Unable to run research workflow: {exc}")
            return
    
        st.subheader(f"{selected_strategy} Research")
        if selected_result is not None:
            metric_cols = st.columns(5)
            metric_cols[0].metric("Total return", format_percent(selected_result.metrics["total_return"]))
            metric_cols[1].metric("Annual return", format_percent(selected_result.metrics["annualized_return"]))
            metric_cols[2].metric("Annual vol", format_percent(selected_result.metrics["annualized_volatility"]))
            metric_cols[3].metric("Sharpe", f"{selected_result.metrics['sharpe_ratio']:.2f}")
            metric_cols[4].metric("Max drawdown", format_percent(selected_result.metrics["max_drawdown"]))
            if selected_strategy == STRATEGY_MOMENTUM:
                st.caption(
                    "Momentum trading buys the strongest recent performers that remain above a longer trend filter, "
                    "then exits when momentum or trend weakens."
                )
            st.line_chart(selected_result.equity_curve.rename(selected_strategy))
        elif swing_result is not None:
            swing_metric_cols = st.columns(5)
            swing_metric_cols[0].metric("CAGR", format_percent(swing_result.metrics["annualized_return"]))
            swing_metric_cols[1].metric("Total return", format_percent(swing_result.metrics["total_return"]))
            swing_metric_cols[2].metric("Max drawdown", format_percent(swing_result.metrics["max_drawdown"]))
            swing_metric_cols[3].metric("Sharpe", f"{swing_result.metrics['sharpe_ratio']:.2f}")
            swing_metric_cols[4].metric("Win rate", format_percent(swing_result.metrics["win_rate"]))
            comparison = pd.DataFrame(
                {
                    "Swing strategy": swing_result.equity_curve,
                    "SPY": swing_result.benchmark_equity_curve,
                }
            ).ffill()
            st.line_chart(comparison)
            st.dataframe(
                pd.DataFrame(
                    [
                        {"benchmark": "Swing strategy", **swing_result.metrics},
                        {"benchmark": "SPY", **swing_result.benchmark_metrics},
                    ]
                ).assign(
                    total_return=lambda frame: frame["total_return"].map(format_percent),
                    cagr=lambda frame: frame["cagr"].map(format_percent),
                    annualized_return=lambda frame: frame["annualized_return"].map(format_percent),
                    max_drawdown=lambda frame: frame["max_drawdown"].map(format_percent),
                    annualized_volatility=lambda frame: frame["annualized_volatility"].map(format_percent),
                    win_rate=lambda frame: frame["win_rate"].map(format_percent),
                )[
                    [
                        "benchmark",
                        "cagr",
                        "total_return",
                        "max_drawdown",
                        "sharpe_ratio",
                        "win_rate",
                        "annualized_volatility",
                    ]
                ],
                use_container_width=True,
            )
            st.caption(
                "Reports saved to "
                + ", ".join(str(path) for path in swing_result.report_paths.values())
            )
            st.subheader("Swing Trades")
            if swing_result.trades.empty:
                st.info("No swing strategy trades were generated for this period.")
            else:
                st.dataframe(swing_result.trades.tail(50), use_container_width=True)
        else:
            st.info("No research result was generated.")
    
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Latest Target Weights")
            st.caption(f"Using {selected_strategy} for the paper order preview.")
            latest_weights = latest_target_weights[latest_target_weights > 0].sort_values(ascending=False)
            if latest_weights.empty:
                st.info("Strategy is currently in cash.")
            else:
                st.dataframe(
                    latest_weights.rename("target_weight").map(format_percent),
                    use_container_width=True,
                )
    
        with right:
            st.subheader("Paper Order Preview")
            if not proposed_orders:
                st.info("No paper orders generated for the latest target.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "ticker": order.ticker,
                                "side": order.side.value,
                                "quantity": order.quantity,
                            }
                            for order in proposed_orders
                        ]
                    ),
                    use_container_width=True,
                )
    
        st.subheader("Alpaca Paper Trading")
        st.caption(
            "Paper endpoint only. Live trading endpoints are not supported. "
            "Manual approval is required before any order submission."
        )
        if alpaca_account is None:
            st.warning(
                "Alpaca paper account is not connected. Add valid paper credentials to .env to fetch "
                "balances, positions, and enable approved paper-order submission."
            )
        else:
            alpaca_cols = st.columns(4)
            alpaca_cols[0].metric("Account status", alpaca_account.status)
            alpaca_cols[1].metric("Cash", f"${alpaca_account.cash:,.2f}")
            alpaca_cols[2].metric("Portfolio value", f"${alpaca_account.portfolio_value:,.2f}")
            alpaca_cols[3].metric("Buying power", f"${alpaca_account.buying_power:,.2f}")
            st.metric("DRY_RUN", str(settings.dry_run))
            st.metric("ALPACA_PAPER", str(settings.alpaca_paper))
            st.metric("AUTO_TRADE", str(settings.auto_trade))
    
            if alpaca_positions:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "ticker": position.ticker,
                                "quantity": position.quantity,
                                "market_value": position.market_value,
                                "average_entry_price": position.average_entry_price,
                            }
                            for position in alpaca_positions
                        ]
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No current Alpaca paper positions.")
    
            approval_disabled = not proposed_orders
            if st.button(
                "Approve and submit paper orders",
                type="primary",
                disabled=approval_disabled,
                use_container_width=True,
            ):
                validate_orders_against_risk_limits(
                    cash=broker.cash,
                    positions=broker.positions,
                    prices=current_prices,
                    orders=proposed_orders,
                    risk_limits=HARD_ORDER_RISK_LIMITS,
                )
                open_orders = alpaca_client.fetch_open_orders()
                latest_price_time = prices.index[-1].to_pydatetime()
                price_timestamps = {
                    ticker: latest_price_time
                    for ticker in current_prices.index
                }
                execution_config = OrderExecutionConfig()
                execution_orders = [
                    build_execution_order(
                        ticker=order.ticker,
                        side=order.side,
                        current_price=float(current_prices[order.ticker]),
                        account_equity=alpaca_account.portfolio_value,
                        existing_quantity=broker.positions.get(order.ticker, 0),
                        configured_quantity=order.quantity,
                        stop_price=float(current_prices[order.ticker]) * 0.95 if order.side == OrderSide.BUY else None,
                        config=execution_config,
                    )
                    for order in proposed_orders
                    if order.ticker in current_prices
                ]
                stop_prices = {
                    order.ticker: float(current_prices[order.ticker]) * 0.95
                    for order in execution_orders
                    if order.side == OrderSide.BUY and order.ticker in current_prices
                }
                pre_trade_results = validate_pre_trade_orders(
                    execution_orders,
                    account_equity=alpaca_account.portfolio_value,
                    positions=broker.positions,
                    prices=current_prices,
                    price_timestamps=price_timestamps,
                    open_orders=open_orders,
                    auto_trade=settings.auto_trade,
                    dry_run=settings.dry_run,
                    manual_confirmed=True,
                    stop_prices=stop_prices,
                )
                st.dataframe(validation_results_to_dataframe(pre_trade_results), use_container_width=True)
                orders_to_submit = accepted_orders(execution_orders, pre_trade_results)
                if not orders_to_submit:
                    st.warning("No orders passed pre-trade risk validation.")
                    return
                submitted = execute_orders(
                    alpaca_client,
                    orders_to_submit,
                    dry_run=settings.dry_run,
                )
                st.success(
                    "Dry-run logged approved orders."
                    if settings.dry_run
                    else "Submitted approved orders to Alpaca paper trading."
                )
                submitted_frame = pd.DataFrame([result.to_dict() for result in submitted])
                st.session_state[SUBMITTED_ORDERS_KEY] = submitted_frame
                st.dataframe(submitted_frame, use_container_width=True)
    
        st.subheader("Price Data")
        st.dataframe(prices.tail(20), use_container_width=True)
    
    
if __name__ == "__main__":
    main()
