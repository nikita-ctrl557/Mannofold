"""Regime-long-only strategy: directional stance driven by regime drift, LONG ONLY.

Like regime_rotation but negative target weights are clamped to zero — the
strategy sits in cash rather than shorting.  Suits assets with upward drift
or portfolios where shorting is operationally undesirable.

Entry/exit hysteresis (same thresholds as regime_rotation) prevents whipsaw on
noisy regime transitions.
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

NAME = "regime_long_only"
DESCRIPTION = (
    "Regime-drift directional sizing with hysteresis, clamped to [0, 1] "
    "(long or flat only — negative weights become cash)."
)

# Tunable knobs — kept consistent with regime_rotation defaults
_GAIN = 3.0            # amplifier inside tanh( gain * mean / std )
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_ENTRY_THRESH = 0.10   # minimum weight to open a long position
_EXIT_THRESH = 0.05    # hold until weight drops below this


class RegimeLongOnlyStrategy:
    """Regime-drift strategy that is long or flat; never short."""

    def __init__(self) -> None:
        # per-symbol: 1 = long, 0 = flat  (no -1 since long-only)
        self._stance: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: tanh-shaped, confidence-gated, then clamped to [0, 1]
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence
        # Long-only: negative raw -> 0 (sit in cash)
        raw = max(0.0, raw)

        # Dead-band: collapse tiny weights to zero
        if raw < _DEAD_BAND:
            raw = 0.0

        current_stance = self._stance.get(sym, 0)

        if current_stance == 0:
            # No position — only enter if conviction clears entry threshold
            if raw >= _ENTRY_THRESH:
                self._stance[sym] = 1
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            # Holding long — exit if conviction decays below exit threshold
            if raw < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
            # Otherwise continue holding; weight scaled by current raw magnitude

        # Emit weight clamped to [0, 1]
        weight = min(1.0, raw)
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh RegimeLongOnlyStrategy instance."""
    return RegimeLongOnlyStrategy()
