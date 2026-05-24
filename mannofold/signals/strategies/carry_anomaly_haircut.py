"""Carry-Anomaly Haircut strategy.

Combines inverse-variance carry sizing (from vol_scaled_carry) with the
smooth linear anomaly haircut (from anomaly_defensive).

Sizing: kelly = fwd_return_mean / (fwd_return_std**2 + eps), clamped.
Base:   tanh(gain * kelly).
Haircut: max(0, 1 - anomaly_score / ANOMALY_CAP)  — decays to 0 before
         full anomaly.
Final:  target_weight = base * haircut * confidence, dead-banded and clamped.
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

NAME = "carry_anomaly_haircut"
DESCRIPTION = (
    "Inverse-variance Kelly carry sizing with a linear anomaly haircut; "
    "exposure decays smoothly to zero before full anomaly regime."
)

_EPS = 1e-9
# Clamp on the raw Kelly ratio to prevent tanh saturation when std is tiny.
_RAW_CLAMP = 10.0
# Gain applied inside tanh squash.
_GAIN = 3.0
# Anomaly score at which the haircut reaches zero (full de-gross).
_ANOMALY_CAP = 0.6
# Dead-band: weights smaller than this are zeroed to suppress noise trading.
_DEADBAND = 0.04


class CarryAnomalyHaircutStrategy:
    """Inverse-variance carry with linear anomaly haircut and confidence gate.

    Mechanics
    ---------
    1. Compute Kelly-like weight: fwd_return_mean / (fwd_return_std**2 + eps).
    2. Clamp to avoid saturation, then squash via tanh(gain * kelly).
    3. Multiply by a LINEAR haircut = max(0, 1 - anomaly_score / anomaly_cap).
    4. Gate by confidence = regime_prob (pure regime stability, no double-counting
       of anomaly since the haircut already handles it).
    5. Apply dead-band, then clamp to [-1, 1].
    """

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_cap: float = _ANOMALY_CAP,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._anomaly_cap = anomaly_cap
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        # Inverse-variance Kelly sizing.
        variance = state.fwd_return_std ** 2 + _EPS
        kelly = state.fwd_return_mean / variance
        kelly = max(-_RAW_CLAMP, min(_RAW_CLAMP, kelly))
        momentum = math.tanh(self._gain * kelly)

        # Confidence: pure regime probability — anomaly handled via haircut in target.
        confidence = max(0.0, min(1.0, state.regime_prob))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard flat on anomalous regime sentinel.
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Linear anomaly haircut: decays to 0 as anomaly_score -> anomaly_cap.
        haircut = max(0.0, 1.0 - signals.anomaly / self._anomaly_cap)

        # Final weight: carry signal * haircut * confidence.
        weight = signals.momentum * haircut * signals.confidence

        # Dead-band: suppress noisy near-zero positions.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh CarryAnomalyHaircutStrategy with default parameters."""
    return CarryAnomalyHaircutStrategy()
