from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor

import pandas as pd


class TradeAction(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass(frozen=True)
class SwingStrategyConfig:
    trend_window: int = 200
    momentum_window: int = 50
    rsi_window: int = 14
    atr_window: int = 14
    min_rsi: float = 40.0
    max_rsi: float = 65.0
    atr_stop_multiple: float = 2.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 5

    def validate(self) -> None:
        if self.trend_window < 1:
            raise ValueError("trend_window must be at least 1")
        if self.momentum_window < 2:
            raise ValueError("momentum_window must be at least 2")
        if self.rsi_window < 1:
            raise ValueError("rsi_window must be at least 1")
        if self.atr_window < 1:
            raise ValueError("atr_window must be at least 1")
        if not 0 < self.min_rsi < self.max_rsi < 100:
            raise ValueError("RSI bounds must satisfy 0 < min_rsi < max_rsi < 100")
        if self.atr_stop_multiple <= 0:
            raise ValueError("atr_stop_multiple must be positive")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")


@dataclass(frozen=True)
class PositionState:
    quantity: int = 0
    entry_price: float | None = None
    stop_loss: float | None = None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True)
class SwingSignal:
    action: TradeAction
    reason: str
    close: float
    sma_200: float | None
    sma_50: float | None
    rsi: float | None
    atr: float | None
    stop_loss: float | None = None
    shares: int = 0
    risk_amount: float = 0.0


def generate_swing_signal(
    ohlc: pd.DataFrame,
    account_equity: float,
    open_positions: int,
    position: PositionState | None = None,
    config: SwingStrategyConfig | None = None,
) -> SwingSignal:
    """Generate a conservative daily swing-trading signal for one ticker."""
    cfg = config or SwingStrategyConfig()
    cfg.validate()
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")

    data = normalize_ohlc(ohlc)
    indicators = compute_indicators(data, cfg)
    latest = indicators.iloc[-1]
    current_position = position or PositionState()

    close = float(latest["Close"])
    atr = _optional_float(latest["atr"])
    active_stop = _active_stop(current_position, close, atr, cfg)
    snapshot = _signal_snapshot(latest, active_stop)

    if current_position.is_open:
        sell_reason = _sell_reason(latest, active_stop, cfg)
        if sell_reason:
            return SwingSignal(action=TradeAction.SELL, reason=sell_reason, **snapshot)
        return SwingSignal(action=TradeAction.HOLD, reason="Open position still satisfies swing filters", **snapshot)

    buy_blocker = _buy_blocker(latest, open_positions, cfg)
    if buy_blocker:
        return SwingSignal(action=TradeAction.HOLD, reason=buy_blocker, **snapshot)

    stop_loss = close - cfg.atr_stop_multiple * float(latest["atr"])
    risk_amount = account_equity * cfg.risk_per_trade
    risk_per_share = close - stop_loss
    shares = floor(risk_amount / risk_per_share) if risk_per_share > 0 else 0
    if shares < 1:
        return SwingSignal(
            action=TradeAction.HOLD,
            reason="Account equity is too small for 1% risk sizing",
            **snapshot,
        )

    return SwingSignal(
        action=TradeAction.BUY,
        reason="Price above 200-day MA, 50-day MA rising, RSI between 40 and 65",
        stop_loss=stop_loss,
        shares=shares,
        risk_amount=risk_amount,
        close=close,
        sma_200=_optional_float(latest["sma_200"]),
        sma_50=_optional_float(latest["sma_50"]),
        rsi=_optional_float(latest["rsi"]),
        atr=atr,
    )


def normalize_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"High", "Low", "Close"}
    missing = required_columns - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC data is missing columns: {', '.join(sorted(missing))}")
    if ohlc.empty:
        raise ValueError("OHLC data is empty")
    return ohlc.sort_index().loc[:, ["High", "Low", "Close"]].dropna()


def compute_indicators(ohlc: pd.DataFrame, config: SwingStrategyConfig) -> pd.DataFrame:
    data = ohlc.copy()
    data["sma_200"] = data["Close"].rolling(
        config.trend_window,
        min_periods=config.trend_window,
    ).mean()
    data["sma_50"] = data["Close"].rolling(
        config.momentum_window,
        min_periods=config.momentum_window,
    ).mean()
    data["sma_50_rising"] = data["sma_50"] > data["sma_50"].shift(1)
    data["rsi"] = relative_strength_index(data["Close"], config.rsi_window)
    data["atr"] = average_true_range(data, config.atr_window)
    return data


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window, min_periods=window).mean()
    average_loss = losses.rolling(window, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(average_loss != 0, 100.0)


def average_true_range(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = ohlc["Close"].shift(1)
    true_range = pd.concat(
        [
            ohlc["High"] - ohlc["Low"],
            (ohlc["High"] - previous_close).abs(),
            (ohlc["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def _buy_blocker(latest: pd.Series, open_positions: int, config: SwingStrategyConfig) -> str | None:
    missing = _missing_indicator_reason(latest)
    if missing:
        return missing
    if open_positions >= config.max_open_positions:
        return "Max open positions reached"
    if latest["Close"] <= latest["sma_200"]:
        return "Price is not above 200-day moving average"
    if not bool(latest["sma_50_rising"]):
        return "50-day moving average is not rising"
    if not config.min_rsi <= latest["rsi"] <= config.max_rsi:
        return "RSI is outside 40 to 65 range"
    if latest["atr"] <= 0:
        return "ATR is not positive"
    return None


def _sell_reason(latest: pd.Series, stop_loss: float | None, config: SwingStrategyConfig) -> str | None:
    missing = _missing_indicator_reason(latest)
    if missing:
        return None
    if stop_loss is not None and latest["Close"] <= stop_loss:
        return "Price hit 2 ATR stop loss"
    if latest["Close"] <= latest["sma_200"]:
        return "Price fell below 200-day moving average"
    if not bool(latest["sma_50_rising"]):
        return "50-day moving average stopped rising"
    if not config.min_rsi <= latest["rsi"] <= config.max_rsi:
        return "RSI moved outside 40 to 65 range"
    return None


def _missing_indicator_reason(latest: pd.Series) -> str | None:
    required = ["sma_200", "sma_50", "rsi", "atr"]
    if latest[required].isna().any():
        return "Not enough price history for swing strategy indicators"
    return None


def _active_stop(
    position: PositionState,
    close: float,
    atr: float | None,
    config: SwingStrategyConfig,
) -> float | None:
    if position.stop_loss is not None:
        return position.stop_loss
    if position.entry_price is not None and atr is not None:
        return position.entry_price - config.atr_stop_multiple * atr
    if position.is_open and atr is not None:
        return close - config.atr_stop_multiple * atr
    return None


def _signal_snapshot(latest: pd.Series, stop_loss: float | None) -> dict[str, float | None]:
    return {
        "close": float(latest["Close"]),
        "sma_200": _optional_float(latest["sma_200"]),
        "sma_50": _optional_float(latest["sma_50"]),
        "rsi": _optional_float(latest["rsi"]),
        "atr": _optional_float(latest["atr"]),
        "stop_loss": stop_loss,
    }


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
