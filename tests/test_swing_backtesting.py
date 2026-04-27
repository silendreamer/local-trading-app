from __future__ import annotations

import pandas as pd

from trading_app.backtesting import CostAssumptions, run_swing_strategy_backtest
from trading_app.strategies.swing_strategy import SwingStrategyConfig


def make_ohlc(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, index=pd.date_range("2024-01-01", periods=len(closes)))
    return pd.DataFrame(
        {
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        }
    )


def test_swing_backtest_runs_across_tickers_and_saves_reports(tmp_path) -> None:
    strategy_data = {
        "AAA": make_ohlc([100, 101, 102, 103, 102, 103, 103, 103]),
        "BBB": make_ohlc([50, 50.5, 51, 51.5, 51, 51.5, 51.5, 51.5]),
    }
    spy_data = make_ohlc([100, 100, 101, 102, 103, 104, 105, 106])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
        max_open_positions=5,
    )

    result = run_swing_strategy_backtest(
        strategy_data,
        spy_data,
        initial_cash=10_000.0,
        config=config,
        costs=CostAssumptions(slippage_bps=10, transaction_cost_bps=5),
        reports_dir=tmp_path,
    )

    assert not result.equity_curve.empty
    assert not result.benchmark_equity_curve.empty
    assert {"total_return", "annualized_return", "max_drawdown", "sharpe_ratio", "win_rate"} <= set(result.metrics)
    assert result.metrics["win_rate"] >= 0.0
    assert not result.trades.empty
    assert result.report_paths["equity"].exists()
    assert result.report_paths["metrics"].exists()
    assert result.report_paths["trades"].exists()


def test_swing_backtest_includes_cost_drag(tmp_path) -> None:
    strategy_data = {"AAA": make_ohlc([100, 101, 102, 103, 102, 103, 103, 103])}
    spy_data = make_ohlc([100, 100, 101, 102, 103, 104, 105, 106])
    config = SwingStrategyConfig(
        trend_window=5,
        momentum_window=3,
        rsi_window=4,
        atr_window=3,
    )

    no_cost = run_swing_strategy_backtest(
        strategy_data,
        spy_data,
        initial_cash=10_000.0,
        config=config,
        costs=CostAssumptions(slippage_bps=0, transaction_cost_bps=0),
        reports_dir=tmp_path / "no_cost",
    )
    with_cost = run_swing_strategy_backtest(
        strategy_data,
        spy_data,
        initial_cash=10_000.0,
        config=config,
        costs=CostAssumptions(slippage_bps=25, transaction_cost_bps=25),
        reports_dir=tmp_path / "with_cost",
    )

    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]
