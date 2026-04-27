from __future__ import annotations

import sys
from base64 import b64encode
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
    OrderRiskLimits,
    PaperBroker,
    cap_target_weights,
    log_proposed_orders,
    orders_for_target_weights,
    validate_orders_against_risk_limits,
)
from trading_app.config import load_environment, load_tickers
from trading_app.data import MarketDataRequest, fetch_ohlc, latest_prices
from trading_app.logging_config import configure_logging
from trading_app.risk import RiskLimits, apply_risk_limits
from trading_app.strategies.momentum_strategy import MomentumTradingConfig, momentum_trading_signals
from trading_app.strategies.strategy import MovingAverageConfig, moving_average_signals, target_equal_weights
from trading_app.strategies.swing_strategy import SwingStrategyConfig


st.set_page_config(page_title="Paper Trading Research", layout="wide")
HARD_ORDER_RISK_LIMITS = OrderRiskLimits()
HEADER_IMAGE = ROOT / "assets" / "trading-header.png"
STRATEGY_MOVING_AVERAGE = "Moving average crossover"
STRATEGY_MOMENTUM = "Momentum trading"
STRATEGY_SWING = "Swing strategy"
STRATEGIES = [STRATEGY_MOVING_AVERAGE, STRATEGY_MOMENTUM, STRATEGY_SWING]


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


def main() -> None:
    settings = load_environment()
    configure_logging(settings.log_level)
    tickers = load_tickers()

    render_header()
    home_tab, trade_tab = st.tabs(["HOME", "TRADE"])
    with home_tab:
        render_home(settings)

    with trade_tab:
        st.info("Use the Home tab to choose a strategy, run research, and review the latest paper order preview.")

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
                proposed_orders = orders_for_target_weights(
                    cash=broker.cash,
                    positions=broker.positions,
                    prices=current_prices,
                    target_weights=latest_target_weights,
                    risk_limits=HARD_ORDER_RISK_LIMITS,
                )
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
                submitted = [
                    alpaca_client.submit_order(order, approved=True)
                    for order in proposed_orders
                ]
                st.success(
                    "Dry-run logged approved orders."
                    if settings.dry_run
                    else "Submitted approved orders to Alpaca paper trading."
                )
                st.dataframe(pd.DataFrame(submitted), use_container_width=True)
    
        st.subheader("Price Data")
        st.dataframe(prices.tail(20), use_container_width=True)
    
    
if __name__ == "__main__":
    main()
