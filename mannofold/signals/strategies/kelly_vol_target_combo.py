"""Kelly vol-target combo strategy.

Combines Kelly criterion for DIRECTION with explicit constant-risk vol targeting
for MAGNITUDE at the signal level:

  direction      = sign(kelly) = sign(fwd_return_mean / (fwd_return_std^2 + eps))
  vol_target_mag = tanh(gain * target_risk / (fwd_return_std + eps))
  target_weight  = direction * vol_target_mag * confidence

where confidence = regime_prob * (1 - anomaly_score).

Volatile neighbourhoods receive smaller size; calm ones receive larger size.
Flat on ANOMALY_REGIME or anomaly_score > anomaly_gate.  Dead-band suppresses
noise trades with |weight| < dead_band.
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

NAME = "kelly_vol_target_combo"
DESCRIPTION = (
    "Kelly direction (sign of mu/sigma^2) combined with constant-risk vol-targeting "
    "magnitude (tanh(gain * target_risk / sigma)); confidence-gated; flat on anomaly."
)

_EPS = 1e-9
_TARGET_RISK = 0.02    # desired risk per unit — scales the tanh argument
_GAIN = 2.0            # amplifier inside tanh magnitude
_ANOMALY_GATE = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04      # collapse |weight| below this to 0


class KellyVolTargetComboStrategy:
    """Combines Kelly direction with explicit vol-targeting magnitude."""

    def __init__(
        self,
        target_risk: float = _TARGET_RISK,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        dead_band: float = _DEAD_BAND,
    ) -> None:
        self._target_risk = target_risk
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._dead_band = dead_band

    def signals(self, state: ManifoldState) -> SignalSet:
        # Kelly ratio: mu / sigma^2; take sign for direction only.
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance
        direction = math.copysign(1.0, kelly_raw) if kelly_raw != 0.0 else 0.0

        # Vol-targeting magnitude: tanh(gain * target_risk / vol).
        vol_target_mag = math.tanh(
            self._gain * self._target_risk / (state.fwd_return_std + _EPS)
        )

        # Momentum = direction * vol-targeting magnitude.
        momentum = direction * vol_target_mag

        # Confidence: regime stability attenuated by anomaly.
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

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
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: vol-targeted direction scaled by confidence.
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small, noisy weights.
        if abs(raw) < self._dead_band:
            raw = 0.0

        # Clamp to [-1, 1].
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh KellyVolTargetComboStrategy instance."""
    return KellyVolTargetComboStrategy()
