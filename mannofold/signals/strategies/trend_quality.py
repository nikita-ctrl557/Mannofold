"""Trend-quality strategy: regime drift sizing gated by density typicality + hysteresis.

Evolutionary cross of regime_rotation (directional stance + hysteresis) and
density_gated (manifold typicality gate). Commits to a directional position only
when the Sharpe-normalised drift is both LARGE (clears entry threshold) and
HIGH QUALITY (density gate passes). Hysteresis prevents whipsaw; confidence fuses
regime stability with anomaly penalty. Flat on ANOMALY_REGIME or high anomaly.
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

NAME = "trend_quality"
DESCRIPTION = "Regime-drift sizing gated by density typicality and entry/exit hysteresis for high-quality trend commitment."

# --------------------------------------------------------------------------- #
#  Tunable knobs                                                               #
# --------------------------------------------------------------------------- #
_GAIN = 3.0            # tanh amplifier: tanh(gain * sharpe)
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_ENTRY_THRESH = 0.10   # minimum |raw * density_gate * confidence| to enter
_EXIT_THRESH = 0.05    # hold until |raw * density_gate * confidence| < this
_DENSITY_MID = 1.0     # density value at which density gate = 0.5
_DENSITY_SCALE = 2.0   # sigmoid steepness for density gate
_DENSITY_CLAMP = 50.0  # defensive upper clamp on raw density values


# --------------------------------------------------------------------------- #
#  Density gate helper (borrowed from density_gated)                           #
# --------------------------------------------------------------------------- #
def _density_gate(density: float) -> float:
    """Smooth sigmoid gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


# --------------------------------------------------------------------------- #
#  Strategy                                                                    #
# --------------------------------------------------------------------------- #
class TrendQualityStrategy:
    """Directional sizing committed only in high-quality, high-density regimes.

    Signal pipeline:
      sharpe  = fwd_return_mean / (fwd_return_std + eps)
      raw     = tanh(gain * sharpe)
      gate    = sigmoid density gate in [0, 1]
      confidence = regime_prob * (1 - anomaly_score)
      weight  = raw * gate * confidence
    Hysteresis (per-symbol):
      enter only when |weight| >= ENTRY_THRESH
      hold   until  |weight| <  EXIT_THRESH
    """

    def __init__(self) -> None:
        # per-symbol signed stance: +1, -1, or 0
        self._stance: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
        gate = _density_gate(state.density)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,                    # raw Sharpe; target() applies tanh(gain * .)
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,       # density quality folded into confidence
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or excessive anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Proposed raw weight: tanh-compressed Sharpe scaled by quality confidence
        # signals.momentum == sharpe; signals.confidence == base_conf * density_gate
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: treat negligible weights as zero
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        current_stance = self._stance.get(sym, 0)
        desired_sign = 1 if raw > 0 else (-1 if raw < 0 else 0)

        if current_stance == 0:
            # No position — only enter if quality-gated conviction clears entry threshold
            if abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            # Holding — flip side only if new direction clears entry threshold
            if desired_sign != 0 and desired_sign != current_stance and abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            # Exit if quality-gated conviction decays below exit threshold
            elif abs(raw) < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
            # Otherwise hold current stance; weight magnitude updated below

        # Emit weight aligned to current stance, clamped to [-1, 1]
        weight = max(-1.0, min(1.0, abs(raw) * self._stance[sym]))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh TrendQualityStrategy instance."""
    return TrendQualityStrategy()
