from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd

from trading_app.broker import OrderRiskLimits
from trading_app.strategies.momentum_strategy import MomentumTradingConfig, momentum_scores
from trading_app.strategies.swing_strategy import average_true_range


class TradeDecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


@dataclass(frozen=True)
class IntradayDecisionConfig:
    momentum_lookback: int = 4
    trend_window: int = 8
    min_momentum_return: float = 0.003
    max_vwap_extension: float = 0.015
    max_ma_extension: float = 0.02
    max_spread_pct: float = 0.012
    atr_window: int = 4
    max_atr_pct: float = 0.035
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    trailing_stop_pct: float = 0.025

    def validate(self) -> None:
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be at least 1")
        if self.trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if self.min_momentum_return < 0:
            raise ValueError("min_momentum_return cannot be negative")
        if self.max_vwap_extension < 0:
            raise ValueError("max_vwap_extension cannot be negative")
        if self.max_ma_extension < 0:
            raise ValueError("max_ma_extension cannot be negative")
        if self.max_spread_pct < 0:
            raise ValueError("max_spread_pct cannot be negative")
        if self.atr_window < 1:
            raise ValueError("atr_window must be at least 1")
        if self.max_atr_pct < 0:
            raise ValueError("max_atr_pct cannot be negative")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.trailing_stop_pct <= 0:
            raise ValueError("trailing_stop_pct must be positive")


@dataclass(frozen=True)
class IntradayPositionState:
    quantity: int
    entry_price: float
    highest_price_since_entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True)
class TradeDecision:
    ticker: str
    action: TradeDecisionAction
    reason: str
    current_price: float | None = None
    trend_value: float | None = None
    momentum_return: float | None = None
    vwap: float | None = None
    spread_pct: float | None = None
    atr_pct: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


def decide_intraday_trade(
    ticker: str,
    ohlc: pd.DataFrame,
    *,
    positions: dict[str, int],
    account_equity: float,
    risk_limits: OrderRiskLimits | None = None,
    position_state: IntradayPositionState | None = None,
    config: IntradayDecisionConfig | None = None,
) -> TradeDecision:
    cfg = config or IntradayDecisionConfig()
    cfg.validate()
    limits = risk_limits or OrderRiskLimits()
    limits.validate()

    data = _normalize_intraday_data(ohlc)
    if data is None:
        return TradeDecision(ticker=ticker, action=TradeDecisionAction.SKIP, reason="Missing complete intraday OHLC data")
    if account_equity <= 0:
        return TradeDecision(ticker=ticker, action=TradeDecisionAction.SKIP, reason="Account equity must be positive")

    owned_quantity = positions.get(ticker, 0)
    current_position = position_state
    if current_position is None and owned_quantity > 0:
        current_position = IntradayPositionState(
            quantity=owned_quantity,
            entry_price=float(data["Close"].iloc[-1]),
        )

    indicators = _compute_decision_indicators(data, cfg)
    if indicators is None:
        return TradeDecision(
            ticker=ticker,
            action=TradeDecisionAction.SKIP,
            reason="Not enough intraday candles for trend, momentum, and volatility checks",
        )

    if current_position is not None and current_position.is_open:
        return _evaluate_exit(ticker, current_position, indicators, cfg)
    return _evaluate_entry(ticker, positions, account_equity, limits, indicators, cfg)


def _evaluate_entry(
    ticker: str,
    positions: dict[str, int],
    account_equity: float,
    risk_limits: OrderRiskLimits,
    indicators: dict[str, float | None],
    config: IntradayDecisionConfig,
) -> TradeDecision:
    active_positions = {symbol: quantity for symbol, quantity in positions.items() if quantity > 0}
    if len(active_positions) >= risk_limits.max_open_positions:
        return _decision(ticker, TradeDecisionAction.HOLD, "Max open positions reached", indicators)

    current_price = _required(indicators, "current_price")
    trend_value = _required(indicators, "trend_value")
    momentum_return = _required(indicators, "momentum_return")
    spread_pct = _required(indicators, "spread_pct")
    atr_pct = _required(indicators, "atr_pct")
    vwap = indicators["vwap"]

    if current_price <= trend_value:
        return _decision(ticker, TradeDecisionAction.HOLD, "Trend condition is not met: price is below moving average", indicators)
    if momentum_return < config.min_momentum_return:
        return _decision(ticker, TradeDecisionAction.HOLD, "Momentum condition is not met", indicators)
    if vwap is not None and current_price > vwap * (1 + config.max_vwap_extension):
        return _decision(ticker, TradeDecisionAction.HOLD, "Price is extended too far above VWAP", indicators)
    if current_price > trend_value * (1 + config.max_ma_extension):
        return _decision(ticker, TradeDecisionAction.HOLD, "Price is extended too far above moving average", indicators)
    if spread_pct > config.max_spread_pct:
        return _decision(ticker, TradeDecisionAction.HOLD, "Intraday candle spread is too high", indicators)
    if atr_pct > config.max_atr_pct:
        return _decision(ticker, TradeDecisionAction.HOLD, "Intraday volatility is too high", indicators)

    max_position_value = account_equity * risk_limits.max_position_weight
    if max_position_value < current_price:
        return _decision(ticker, TradeDecisionAction.HOLD, "Risk limit leaves insufficient buying capacity for one share", indicators)

    return _decision(
        ticker,
        TradeDecisionAction.BUY,
        "Trend and momentum conditions are met within extension, spread, volatility, and risk limits",
        indicators,
    )


def _evaluate_exit(
    ticker: str,
    position: IntradayPositionState,
    indicators: dict[str, float | None],
    config: IntradayDecisionConfig,
) -> TradeDecision:
    current_price = _required(indicators, "current_price")
    trend_value = _required(indicators, "trend_value")
    momentum_return = _required(indicators, "momentum_return")

    stop_loss = position.stop_loss or position.entry_price * (1 - config.stop_loss_pct)
    take_profit = position.take_profit or position.entry_price * (1 + config.take_profit_pct)
    highest_price = max(position.highest_price_since_entry or position.entry_price, current_price)
    trailing_stop = position.trailing_stop or highest_price * (1 - config.trailing_stop_pct)

    if current_price <= stop_loss:
        return _decision(ticker, TradeDecisionAction.SELL, "Stop loss is hit", indicators)
    if current_price <= trailing_stop:
        return _decision(ticker, TradeDecisionAction.SELL, "Trailing stop is hit", indicators)
    if current_price >= take_profit:
        return _decision(ticker, TradeDecisionAction.SELL, "Take profit target is hit", indicators)
    if current_price <= trend_value:
        return _decision(ticker, TradeDecisionAction.SELL, "Strategy exit signal: price fell below moving average", indicators)
    if momentum_return <= 0:
        return _decision(ticker, TradeDecisionAction.SELL, "Strategy exit signal: momentum turned non-positive", indicators)

    return _decision(ticker, TradeDecisionAction.HOLD, "No exit condition is met", indicators)


def _compute_decision_indicators(
    data: pd.DataFrame,
    config: IntradayDecisionConfig,
) -> dict[str, float | None] | None:
    minimum_rows = max(config.trend_window, config.momentum_lookback + 1, config.atr_window + 1)
    if len(data) < minimum_rows:
        return None

    close_prices = data[["Close"]].rename(columns={"Close": "ticker"})
    momentum_config = MomentumTradingConfig(
        lookback_window=config.momentum_lookback,
        trend_window=config.trend_window,
        min_momentum_return=config.min_momentum_return,
        max_positions=1,
    )
    momentum_return = momentum_scores(close_prices, momentum_config)["ticker"].iloc[-1]
    trend_value = data["Close"].rolling(config.trend_window, min_periods=config.trend_window).mean().iloc[-1]
    atr = average_true_range(data[["High", "Low", "Close"]], config.atr_window).iloc[-1]
    latest = data.iloc[-1]
    current_price = float(latest["Close"])
    spread_pct = float((latest["High"] - latest["Low"]) / current_price)
    atr_pct = float(atr / current_price)
    vwap = _vwap(data)

    if pd.isna(momentum_return) or pd.isna(trend_value) or pd.isna(atr):
        return None
    return {
        "current_price": current_price,
        "trend_value": float(trend_value),
        "momentum_return": float(momentum_return),
        "vwap": vwap,
        "spread_pct": spread_pct,
        "atr_pct": atr_pct,
    }


def _normalize_intraday_data(ohlc: pd.DataFrame) -> pd.DataFrame | None:
    required = {"High", "Low", "Close"}
    if ohlc.empty or not required <= set(ohlc.columns):
        return None
    columns = ["High", "Low", "Close"]
    if "Volume" in ohlc.columns:
        columns.append("Volume")
    data = ohlc.loc[:, columns].dropna(subset=["High", "Low", "Close"]).sort_index()
    return data if not data.empty else None


def _vwap(data: pd.DataFrame) -> float | None:
    if "Volume" not in data.columns:
        return None
    volume = data["Volume"].fillna(0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return None
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    return float((typical_price * volume).sum() / total_volume)


def _decision(
    ticker: str,
    action: TradeDecisionAction,
    reason: str,
    indicators: dict[str, float | None],
) -> TradeDecision:
    return TradeDecision(
        ticker=ticker,
        action=action,
        reason=reason,
        current_price=indicators["current_price"],
        trend_value=indicators["trend_value"],
        momentum_return=indicators["momentum_return"],
        vwap=indicators["vwap"],
        spread_pct=indicators["spread_pct"],
        atr_pct=indicators["atr_pct"],
    )


def _required(indicators: dict[str, float | None], key: str) -> float:
    value = indicators[key]
    if value is None:
        raise ValueError(f"{key} is required")
    return value
