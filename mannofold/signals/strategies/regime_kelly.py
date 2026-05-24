"""Regime-Kelly strategy: regime drift stance with hysteresis + Kelly inverse-variance sizing.

Combines directional conviction from regime_long_only (with shorts allowed) and
Kelly inverse-variance magnitude sizing from kelly_capped.

Direction: hysteresis-gated stance (long=+1, flat=0, short=-1) based on regime
confidence. Flips only when conviction clears _ENTRY_THRESH; holds until it drops
below _EXIT_THRESH.

Magnitude: half-Kelly inverse-variance (mu/sigma^2), clamped and tanh-squashed.

target_weight = direction * |tanh(gain * 0.5 * kelly)| * confidence
where confidence = regime_prob * (1 - anomaly_score).

Flat on ANOMALY_REGIME or anomaly_score > _ANOMALY_THRESH.
Dead-band |w| < 0.04 -> 0.
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

NAME = "regime_kelly"
DESCRIPTION = (
    "Regime-drift directional stance (long/flat/short) with entry/exit hysteresis "
    "combined with half-Kelly inverse-variance magnitude sizing and confidence gating."
)

_EPS = 1e-9
_GAIN = 2.0            # tanh gain applied to Kelly ratio
_KELLY_FRACTION = 0.5  # half-Kelly
_KELLY_CAP = 3.0       # hard cap on |kelly * fraction| before tanh
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_ENTRY_THRESH = 0.10   # minimum |weight| to open/flip a position
_EXIT_THRESH = 0.05    # hold until |weight| drops below this


class RegimeKellyStrategy:
    """Combines regime-drift hysteresis (direction) with Kelly sizing (magnitude)."""

    def __init__(self) -> None:
        # per-symbol stance: +1=long, 0=flat, -1=short
        self._stance: dict[str, int] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        # Confidence: regime stability attenuated by anomaly proximity
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Kelly inverse-variance sizing: mu / sigma^2
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance

        # Apply fraction then hard-cap
        kelly_scaled = _KELLY_FRACTION * kelly_raw
        kelly_clamped = max(-_KELLY_CAP, min(_KELLY_CAP, kelly_scaled))

        # Store Kelly value as momentum for use in target()
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=kelly_clamped,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight magnitude: |tanh(gain * kelly)| * confidence
        kelly_magnitude = abs(math.tanh(_GAIN * signals.momentum))
        raw_magnitude = kelly_magnitude * signals.confidence

        # Dead-band: collapse tiny weights to zero
        if raw_magnitude < _DEAD_BAND:
            raw_magnitude = 0.0

        # Direction from sign of kelly (fwd_return_mean drives sign via kelly)
        direction = 1 if signals.momentum >= 0 else -1

        # Signed raw weight
        raw = direction * raw_magnitude

        current_stance = self._stance.get(sym, 0)

        if current_stance == 0:
            # No position — only enter if conviction clears entry threshold
            if raw_magnitude >= _ENTRY_THRESH:
                self._stance[sym] = direction
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            # Holding a position
            if raw_magnitude < _EXIT_THRESH:
                # Conviction too low — exit to flat
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
            elif direction != current_stance and raw_magnitude >= _ENTRY_THRESH:
                # Regime flipped with sufficient conviction — flip stance
                self._stance[sym] = direction
            else:
                # Keep existing stance direction, but update magnitude
                direction = current_stance
                raw = direction * raw_magnitude

        # Emit weight clamped to [-1, 1]
        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh RegimeKellyStrategy instance."""
    return RegimeKellyStrategy()
