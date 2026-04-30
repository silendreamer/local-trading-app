from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_app.broker import AlpacaAccount, AlpacaPosition
from trading_app.config import AppSettings
from trading_app.data import MarketDataRequest
from trading_app.intraday_loop import MARKET_TIMEZONE
from trading_app.trade_cycle import run_trade_cycle


MARKET_TIME = datetime(2026, 4, 27, 11, 30, tzinfo=MARKET_TIMEZONE)


class FakeBroker:
    def __init__(self, positions=None):
        self.positions = positions or []
        self.submitted = []

    def fetch_account(self):
        return AlpacaAccount(
            account_id="account",
            status="ACTIVE",
            cash=100_000.0,
            portfolio_value=100_000.0,
            buying_power=100_000.0,
        )

    def fetch_positions(self):
        return self.positions

    def fetch_open_orders(self):
        return []

    def submit_order(self, order, approved=False):
        self.submitted.append((order, approved))
        return {"id": "order-id", "status": "accepted", "symbol": order.ticker}


def make_ohlc(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, index=pd.date_range("2026-04-27 09:30", periods=len(closes), freq="15min", tz=MARKET_TIMEZONE))
    return pd.DataFrame(
        {
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": 1_000,
        }
    )


def fake_fetcher(request: MarketDataRequest):
    return {"AAPL": make_ohlc([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6])}


def test_trade_cycle_skips_when_market_closed(tmp_path) -> None:
    summary = run_trade_cycle(
        AppSettings(),
        now=datetime(2026, 4, 27, 16, 1, tzinfo=MARKET_TIMEZONE),
        output_dir=tmp_path,
    )

    assert summary.status == "skipped"
    assert summary.reason == "Market is closed"
    assert summary.market_open is False


def test_trade_cycle_dry_run_logs_but_does_not_submit(tmp_path) -> None:
    broker = FakeBroker()

    summary = run_trade_cycle(
        AppSettings(alpaca_api_key="key", alpaca_secret_key="secret", dry_run=True),
        approved=False,
        now=MARKET_TIME,
        ticker_loader=lambda: ["AAPL"],
        market_data_fetcher=fake_fetcher,
        broker_factory=lambda settings: broker,
        output_dir=tmp_path,
    )

    assert summary.status == "completed"
    assert summary.decision_count == 1
    assert summary.proposed_order_count == 1
    assert summary.submitted_order_count == 1
    assert broker.submitted == []
    assert (tmp_path / "latest_scan_results.csv").exists()
    assert (tmp_path / "latest_order_results.csv").exists()


def test_trade_cycle_submits_when_paper_approved_and_not_dry_run(tmp_path) -> None:
    broker = FakeBroker()

    summary = run_trade_cycle(
        AppSettings(alpaca_api_key="key", alpaca_secret_key="secret", dry_run=False, auto_trade=False),
        approved=True,
        now=MARKET_TIME,
        ticker_loader=lambda: ["AAPL"],
        market_data_fetcher=fake_fetcher,
        broker_factory=lambda settings: broker,
        output_dir=tmp_path,
    )

    assert summary.submitted_order_count == 1
    assert len(broker.submitted) == 1
    assert broker.submitted[0][1] is True


def test_trade_cycle_uses_existing_position_for_exit_logic(tmp_path) -> None:
    broker = FakeBroker(
        positions=[
            AlpacaPosition(
                ticker="AAPL",
                quantity=10,
                market_value=1_000.0,
                average_entry_price=105.0,
                current_price=101.6,
            )
        ]
    )

    summary = run_trade_cycle(
        AppSettings(alpaca_api_key="key", alpaca_secret_key="secret", dry_run=True),
        now=MARKET_TIME,
        ticker_loader=lambda: ["AAPL"],
        market_data_fetcher=fake_fetcher,
        broker_factory=lambda settings: broker,
        output_dir=tmp_path,
    )

    assert summary.decision_count == 1
    assert summary.decisions[0]["action"] in {"SELL", "HOLD"}
    assert summary.decisions[0]["action"] != "BUY"
