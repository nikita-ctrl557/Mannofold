"""Risk-parity trend+carry strategy: combines TREND and CARRY at equal risk.

CONCEPT: Two complementary return sources are blended with inverse-volatility
(risk-parity) weights so neither source dominates and combined returns are
steadier month-over-month. A per-symbol EMA tracks each sub-signal's recent
volatility; weights are inversely proportional so equal risk is contributed.

  trend  = tanh(mu / (sig + eps))               — Sharpe-based direction
  carry  = tanh(clamp(mu / (sig^2 + eps)))       — risk-adjusted carry ratio
  w_i    = (1 / vol_i) / sum(1 / vol_j)          — inverse-vol risk-parity weights
  composite = w_trend * trend + w_carry * carry
  weight = tanh(gain * composite) * confidence
  confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6; dead-band |w| < 0.04 -> 0.
No lookahead: all EMA state updated strictly from past observations.
"""

from __future__ import annotations

import math
from typing import Dict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "risk_parity_trend_carry"
DESCRIPTION = (
    "Trend and carry sub-signals blended with inverse-volatility risk-parity weights "
    "for month-over-month consistency via diversified return sources."
)

_EPS            = 1e-9
_CARRY_CLAMP    = 5.0
_GAIN           = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04
_VOL_ALPHA      = 0.10   # EMA decay for sub-signal volatility tracking
_VOL_INIT       = 0.5    # neutral initial vol (equal weights at startup)


class _SymbolState:
    """Per-symbol EMA volatility state for risk-parity weighting."""

    def __init__(self) -> None:
        self.vol_trend: float = _VOL_INIT
        self.vol_carry: float = _VOL_INIT
        self.prev_trend: float = 0.0
        self.prev_carry: float = 0.0

    def update(self, trend: float, carry: float) -> tuple[float, float]:
        """Update EMA vols with squared change; return inverse-vol weights."""
        # Update vol as EMA of squared first-differences (strictly causal)
        d_trend = trend - self.prev_trend
        d_carry = carry - self.prev_carry
        self.vol_trend = (
            _VOL_ALPHA * d_trend ** 2 + (1.0 - _VOL_ALPHA) * self.vol_trend
        )
        self.vol_carry = (
            _VOL_ALPHA * d_carry ** 2 + (1.0 - _VOL_ALPHA) * self.vol_carry
        )
        self.prev_trend = trend
        self.prev_carry = carry

        # Inverse-vol weights, normalised
        iv_trend = 1.0 / (math.sqrt(max(self.vol_trend, _EPS)))
        iv_carry = 1.0 / (math.sqrt(max(self.vol_carry, _EPS)))
        total = iv_trend + iv_carry
        return iv_trend / total, iv_carry / total


class RiskParityTrendCarryStrategy:
    """Risk-parity blend of trend and carry sub-signals."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Trend: Sharpe-like direction
        trend = math.tanh(mu / (sig + _EPS))

        # Carry: risk-adjusted return ratio, clamped to prevent blowup
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)

        # Risk-parity: update per-symbol EMA vols, get inverse-vol weights
        sym_state = self._get_state(state.symbol)
        w_trend, w_carry = sym_state.update(trend, carry)

        composite = w_trend * trend + w_carry * carry

        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=composite,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh RiskParityTrendCarryStrategy instance."""
    return RiskParityTrendCarryStrategy()
