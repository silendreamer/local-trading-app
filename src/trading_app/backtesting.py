from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_app.config import PROJECT_ROOT
from trading_app.risk import annualized_volatility, max_drawdown
from trading_app.strategies.swing_strategy import SwingStrategyConfig, compute_indicators


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, float]


@dataclass(frozen=True)
class CostAssumptions:
    slippage_bps: float = 5.0
    transaction_cost_bps: float = 2.0

    def validate(self) -> None:
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000

    @property
    def transaction_cost_rate(self) -> float:
        return self.transaction_cost_bps / 10_000


@dataclass
class OpenBacktestPosition:
    quantity: int
    entry_price: float
    stop_loss: float
    entry_date: pd.Timestamp


@dataclass(frozen=True)
class SwingBacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    benchmark_equity_curve: pd.Series
    benchmark_returns: pd.Series
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    trades: pd.DataFrame
    report_paths: dict[str, Path]


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Run a daily close-to-close long-only backtest.

    Signals are shifted by one day to avoid using same-close information.
    """
    if prices.empty:
        raise ValueError("Price data is empty")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    aligned_prices, aligned_weights = prices.align(target_weights, join="inner", axis=0)
    aligned_weights = aligned_weights.reindex(columns=aligned_prices.columns).fillna(0.0)
    asset_returns = aligned_prices.pct_change().fillna(0.0)
    investable_weights = aligned_weights.shift(1).fillna(0.0)
    strategy_returns = (investable_weights * asset_returns).sum(axis=1)
    equity_curve = initial_cash * (1.0 + strategy_returns).cumprod()

    metrics = summarize_performance(equity_curve, strategy_returns)
    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=strategy_returns,
        weights=investable_weights,
        metrics=metrics,
    )


def summarize_performance(equity_curve: pd.Series, returns: pd.Series) -> dict[str, float]:
    """Create common backtest metrics."""
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    years = max(len(equity_curve) / 252, 1 / 252)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    ann_vol = annualized_volatility(returns)
    sharpe = float(annualized_return / ann_vol) if ann_vol else 0.0

    return {
        "total_return": total_return,
        "cagr": annualized_return,
        "annualized_return": annualized_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(equity_curve),
    }


def run_swing_strategy_backtest(
    ohlc_by_ticker: dict[str, pd.DataFrame],
    spy_ohlc: pd.DataFrame,
    initial_cash: float = 100_000.0,
    config: SwingStrategyConfig | None = None,
    costs: CostAssumptions | None = None,
    reports_dir: Path | None = None,
) -> SwingBacktestResult:
    """Backtest the conservative swing strategy across all supplied tickers."""
    cfg = config or SwingStrategyConfig()
    cfg.validate()
    cost_assumptions = costs or CostAssumptions()
    cost_assumptions.validate()
    if not ohlc_by_ticker:
        raise ValueError("At least one ticker of OHLC data is required")
    if spy_ohlc.empty:
        raise ValueError("SPY benchmark data is empty")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    indicators_by_ticker = {
        ticker: compute_indicators(frame.sort_index(), cfg)
        for ticker, frame in sorted(ohlc_by_ticker.items())
        if not frame.empty
    }
    if not indicators_by_ticker:
        raise ValueError("No usable OHLC data supplied")

    all_dates = sorted(set().union(*(frame.index for frame in indicators_by_ticker.values())))
    cash = float(initial_cash)
    positions: dict[str, OpenBacktestPosition] = {}
    closed_trade_rows: list[dict[str, object]] = []
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []

    for current_date in all_dates:
        latest_rows = _latest_rows_for_date(indicators_by_ticker, current_date)
        if not latest_rows:
            continue

        marked_equity = _mark_to_market(cash, positions, latest_rows)
        cash = _process_sells(
            current_date=current_date,
            cash=cash,
            positions=positions,
            latest_rows=latest_rows,
            config=cfg,
            costs=cost_assumptions,
            trades=closed_trade_rows,
        )
        marked_equity = _mark_to_market(cash, positions, latest_rows)
        cash = _process_buys(
            current_date=current_date,
            cash=cash,
            equity=marked_equity,
            positions=positions,
            latest_rows=latest_rows,
            config=cfg,
            costs=cost_assumptions,
            trades=closed_trade_rows,
        )

        equity_values.append(_mark_to_market(cash, positions, latest_rows))
        equity_dates.append(pd.Timestamp(current_date))

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="Strategy")
    daily_returns = equity_curve.pct_change().fillna(0.0)
    trades = pd.DataFrame(closed_trade_rows)
    benchmark_equity = _benchmark_buy_and_hold(spy_ohlc, equity_curve.index, initial_cash)
    benchmark_returns = benchmark_equity.pct_change().fillna(0.0)
    metrics = summarize_performance(equity_curve, daily_returns)
    metrics["win_rate"] = _win_rate(trades)
    benchmark_metrics = summarize_performance(benchmark_equity, benchmark_returns)
    benchmark_metrics["win_rate"] = 0.0
    report_paths = save_backtest_reports(
        equity_curve=equity_curve,
        benchmark_equity_curve=benchmark_equity,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        trades=trades,
        reports_dir=reports_dir or PROJECT_ROOT / "reports",
    )

    return SwingBacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        benchmark_equity_curve=benchmark_equity,
        benchmark_returns=benchmark_returns,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        trades=trades,
        report_paths=report_paths,
    )


def save_backtest_reports(
    equity_curve: pd.Series,
    benchmark_equity_curve: pd.Series,
    metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
    trades: pd.DataFrame,
    reports_dir: Path,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    equity_path = reports_dir / "equity_curve.csv"
    metrics_path = reports_dir / "metrics.csv"
    trades_path = reports_dir / "trades.csv"

    pd.DataFrame(
        {
            "strategy": equity_curve,
            "spy": benchmark_equity_curve.reindex(equity_curve.index).ffill(),
        }
    ).to_csv(equity_path, index_label="date")
    pd.DataFrame(
        [
            {"name": "strategy", **metrics},
            {"name": "SPY", **benchmark_metrics},
        ]
    ).to_csv(metrics_path, index=False)
    trades.to_csv(trades_path, index=False)
    return {"equity": equity_path, "metrics": metrics_path, "trades": trades_path}


def _latest_rows_for_date(
    indicators_by_ticker: dict[str, pd.DataFrame],
    current_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for ticker, frame in indicators_by_ticker.items():
        if current_date in frame.index:
            rows[ticker] = frame.loc[current_date]
    return rows


def _process_sells(
    current_date: pd.Timestamp,
    cash: float,
    positions: dict[str, OpenBacktestPosition],
    latest_rows: dict[str, pd.Series],
    config: SwingStrategyConfig,
    costs: CostAssumptions,
    trades: list[dict[str, object]],
) -> float:
    for ticker in list(positions):
        row = latest_rows.get(ticker)
        if row is None:
            continue
        reason = _swing_sell_reason(row, positions[ticker], config)
        if not reason:
            continue

        position = positions.pop(ticker)
        exit_price = float(row["Close"]) * (1 - costs.slippage_rate)
        gross_proceeds = exit_price * position.quantity
        transaction_cost = gross_proceeds * costs.transaction_cost_rate
        cash += gross_proceeds - transaction_cost
        pnl = (exit_price - position.entry_price) * position.quantity - transaction_cost
        trades.append(
            {
                "ticker": ticker,
                "entry_date": position.entry_date.date(),
                "exit_date": pd.Timestamp(current_date).date(),
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "stop_loss": position.stop_loss,
                "pnl": pnl,
                "return": pnl / (position.entry_price * position.quantity),
                "reason": reason,
            }
        )
    return cash


def _process_buys(
    current_date: pd.Timestamp,
    cash: float,
    equity: float,
    positions: dict[str, OpenBacktestPosition],
    latest_rows: dict[str, pd.Series],
    config: SwingStrategyConfig,
    costs: CostAssumptions,
    trades: list[dict[str, object]],
) -> float:
    candidates = [
        (ticker, row)
        for ticker, row in sorted(latest_rows.items())
        if ticker not in positions and _is_swing_buy(row, config)
    ]
    candidates.sort(key=lambda item: (float(item[1]["rsi"]), item[0]))

    for ticker, row in candidates:
        if len(positions) >= config.max_open_positions:
            break
        close = float(row["Close"])
        atr = float(row["atr"])
        stop_loss = close - config.atr_stop_multiple * atr
        risk_per_share = close - stop_loss
        if risk_per_share <= 0:
            continue

        risk_amount = equity * config.risk_per_trade
        quantity = int(risk_amount // risk_per_share)
        entry_price = close * (1 + costs.slippage_rate)
        estimated_cost = entry_price * quantity * (1 + costs.transaction_cost_rate)
        if quantity < 1 or estimated_cost > cash:
            quantity = int(cash // (entry_price * (1 + costs.transaction_cost_rate)))
        if quantity < 1:
            continue

        gross_cost = entry_price * quantity
        transaction_cost = gross_cost * costs.transaction_cost_rate
        cash -= gross_cost + transaction_cost
        positions[ticker] = OpenBacktestPosition(
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            entry_date=pd.Timestamp(current_date),
        )
        trades.append(
            {
                "ticker": ticker,
                "entry_date": pd.Timestamp(current_date).date(),
                "exit_date": None,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": None,
                "stop_loss": stop_loss,
                "pnl": -transaction_cost,
                "return": 0.0,
                "reason": "BUY",
            }
        )
    return cash


def _mark_to_market(
    cash: float,
    positions: dict[str, OpenBacktestPosition],
    latest_rows: dict[str, pd.Series],
) -> float:
    value = cash
    for ticker, position in positions.items():
        row = latest_rows.get(ticker)
        if row is not None:
            value += position.quantity * float(row["Close"])
        else:
            value += position.quantity * position.entry_price
    return float(value)


def _is_swing_buy(row: pd.Series, config: SwingStrategyConfig) -> bool:
    required = ["sma_200", "sma_50", "rsi", "atr"]
    if row[required].isna().any():
        return False
    return bool(
        row["Close"] > row["sma_200"]
        and row["sma_50_rising"]
        and config.min_rsi <= row["rsi"] <= config.max_rsi
        and row["atr"] > 0
    )


def _swing_sell_reason(
    row: pd.Series,
    position: OpenBacktestPosition,
    config: SwingStrategyConfig,
) -> str | None:
    required = ["sma_200", "sma_50", "rsi", "atr"]
    if row[required].isna().any():
        return None
    if float(row["Close"]) <= position.stop_loss:
        return "Price hit 2 ATR stop loss"
    if row["Close"] <= row["sma_200"]:
        return "Price fell below 200-day moving average"
    if not bool(row["sma_50_rising"]):
        return "50-day moving average stopped rising"
    if not config.min_rsi <= row["rsi"] <= config.max_rsi:
        return "RSI moved outside 40 to 65 range"
    return None


def _benchmark_buy_and_hold(
    spy_ohlc: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    initial_cash: float,
) -> pd.Series:
    spy_close = spy_ohlc["Close"].sort_index().reindex(target_index).ffill().dropna()
    if spy_close.empty:
        return pd.Series(dtype="float64", name="SPY")
    benchmark = initial_cash * (spy_close / spy_close.iloc[0])
    return benchmark.rename("SPY")


def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "exit_date" not in trades:
        return 0.0
    closed = trades[trades["exit_date"].notna()]
    if closed.empty:
        return 0.0
    return float((closed["pnl"] > 0).mean())
