"""Consistent carry strategy: harvest drift only when it is STABLE month-to-month.

Tracks a per-symbol EMA of fwd_return_mean (the drift) AND an EMA of the
absolute change in that drift (drift stability).  Only takes carry exposure
when drift is positive-expectancy AND the neighbourhood drift has been
stable (low recent absolute change).  Unstable or noisy drift => flat.
This targets steady, repeatable carry rather than chasing momentum.

Signal:  sharpe   = fwd_return_mean / (fwd_return_std + eps)
         base     = tanh(gain * tanh(sharpe))
         stability_gate = 1 / (1 + k * ema_drift_change / (ema_drift_abs + eps))
         confidence = regime_prob * (1 - anomaly_score)
Weight:  base * stability_gate * confidence
Gates:   flat on ANOMALY_REGIME or anomaly_score > 0.6
         drift must be positive-expectancy (ema_drift > 0) to go long
         dead-band |w| < 0.04 -> 0
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "consistent_carry"
DESCRIPTION = (
    "Month-over-month consistent carry: harvest drift only when the EMA of "
    "fwd_return_mean has been stable (low absolute change). Unstable drift "
    "yields a near-zero stability gate; steady carry is fully harvested. "
    "Sized by tanh(gain*tanh(sharpe)) * stability_gate * confidence."
)

_EPS = 1e-9
_GAIN = 2.5           # outer tanh gain applied to inner tanh(sharpe)
_ANOMALY_GATE = 0.6   # anomaly_score above this -> flat
_DEADBAND = 0.04      # |weight| below this -> 0
_ALPHA_DRIFT = 0.10   # EMA alpha for per-symbol drift (slow, month-ish)
_ALPHA_CHANGE = 0.15  # EMA alpha for absolute drift change (faster)
_STABILITY_K = 5.0    # sensitivity of stability gate to relative drift change


class ConsistentCarryStrategy:
    """Carry strategy that gates on drift stability (low month-to-month change)."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean
        self._ema_drift: dict[str, float] = {}
        # Per-symbol EMA of |change in drift EMA| (stability tracker)
        self._ema_change: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        drift = state.fwd_return_mean

        # --- Update per-symbol drift EMA (no lookahead: read prev, then update) ---
        if sym not in self._ema_drift:
            self._ema_drift[sym] = drift
            self._ema_change[sym] = 0.0
            ema_drift = drift
            ema_change = 0.0
        else:
            prev_ema = self._ema_drift[sym]
            ema_drift = prev_ema  # value before this bar (no lookahead)
            new_ema = _ALPHA_DRIFT * drift + (1.0 - _ALPHA_DRIFT) * prev_ema
            abs_change = abs(new_ema - prev_ema)
            prev_change = self._ema_change[sym]
            ema_change = _ALPHA_CHANGE * abs_change + (1.0 - _ALPHA_CHANGE) * prev_change
            self._ema_drift[sym] = new_ema
            self._ema_change[sym] = ema_change

        # --- Stability gate: ~1 when drift change is small relative to drift level ---
        ema_drift_abs = abs(ema_drift) + _EPS
        relative_change = ema_change / ema_drift_abs
        stability_gate = 1.0 / (1.0 + _STABILITY_K * relative_change)

        # --- Sharpe-based carry signal ---
        sharpe = drift / (state.fwd_return_std + _EPS)
        base = math.tanh(_GAIN * math.tanh(sharpe))

        # --- Only go long when drift EMA is positive-expectancy ---
        if ema_drift <= 0.0:
            base = min(base, 0.0)  # allow short but not long on negative drift

        # --- Confidence: regime stability attenuated by anomaly ---
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        raw_weight = base * stability_gate

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=raw_weight,
            expected_return=ema_drift,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trading near zero
        if abs(weight) < _DEADBAND:
            weight = 0.0

        # Hard clip to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ConsistentCarryStrategy instance."""
    return ConsistentCarryStrategy()
