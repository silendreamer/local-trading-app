from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.10
    max_gross_exposure: float = 1.0

    def validate(self) -> None:
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 < self.max_gross_exposure <= 1:
            raise ValueError("max_gross_exposure must be in (0, 1]")


def apply_risk_limits(weights: pd.DataFrame, limits: RiskLimits) -> pd.DataFrame:
    """Cap per-position and gross exposure weights."""
    limits.validate()
    if weights.empty:
        return weights.copy()

    capped = weights.clip(lower=0.0, upper=limits.max_position_weight)
    gross = capped.sum(axis=1)
    scale = (limits.max_gross_exposure / gross).where(gross > limits.max_gross_exposure, 1.0)
    scale = scale.fillna(0.0)
    return capped.mul(scale, axis=0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility for a return series."""
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * periods_per_year**0.5)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute max drawdown as a negative decimal value."""
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())
