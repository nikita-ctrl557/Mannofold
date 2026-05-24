"""Renormalization-group multi-scale drift strategy.

Physics perspective: in the renormalization group (RG) framework, a signal that
survives unchanged under coarse-graining (i.e. looks the same at every scale) is
at a *fixed point* — it is structural, not noise.  We operationalize this idea by
computing the drift EMA at three timescales (fast, mid, slow).  When all three
scales agree in sign (scale-invariant), we have a robust fixed-point signal worth
trading at full size.  When the signs flip across scales, the signal is
scale-dependent (UV/IR mismatch) → noise → flat.

coherence  = sign_agreement × min(|fast|, |mid|, |slow|) / (max + ε)
           (normalized ratio in [0,1]; zero if any two scales disagree in sign)
target_weight = coherent_dir · tanh(GAIN · |tanh(sharpe)| · coherence) · confidence
confidence = regime_prob · (1 − anomaly_score)

Flat on ANOMALY_REGIME, anomaly > 0.6, or scale-incoherent.
Dead-band |w| < 0.04 → 0.
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

NAME = "renorm_multiscale"
DESCRIPTION = (
    "Renormalization-group multi-scale strategy: a drift signal that is "
    "scale-invariant (same sign across fast/mid/slow EMAs) is treated as a robust "
    "fixed point worth trading at full size; scale-dependent signals (sign flips "
    "across scales) are noise and receive zero weight."
)

# EMA decay rates (one per RG scale)
_FAST_ALPHA = 0.5    # fast / UV scale
_MID_ALPHA  = 0.2    # intermediate scale
_SLOW_ALPHA = 0.05   # slow / IR scale

_GAIN           = 2.5   # amplifier inside outer tanh
_ANOMALY_THRESH = 0.6   # anomaly_score above this → flat
_DEAD_BAND      = 0.04  # collapse |weight| below this to 0


class RenormMultiscaleStrategy:
    """Per-symbol three-scale EMA strategy inspired by the renormalization group."""

    def __init__(self) -> None:
        self._fast: dict[str, float] = {}
        self._mid:  dict[str, float] = {}
        self._slow: dict[str, float] = {}
        self._init: dict[str, bool]  = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym   = state.symbol
        drift = state.fwd_return_mean

        # Bootstrap on first observation (no look-ahead)
        if not self._init.get(sym, False):
            self._fast[sym] = drift
            self._mid[sym]  = drift
            self._slow[sym] = drift
            self._init[sym] = True
        else:
            self._fast[sym] = _FAST_ALPHA * drift + (1.0 - _FAST_ALPHA) * self._fast[sym]
            self._mid[sym]  = _MID_ALPHA  * drift + (1.0 - _MID_ALPHA)  * self._mid[sym]
            self._slow[sym] = _SLOW_ALPHA * drift + (1.0 - _SLOW_ALPHA) * self._slow[sym]

        fast = self._fast[sym]
        mid  = self._mid[sym]
        slow = self._slow[sym]

        # RG coherence: all three scales must share the same sign
        signs_agree = (fast * mid > 0) and (mid * slow > 0)
        coherent_dir = math.copysign(1.0, fast) if signs_agree and fast != 0.0 else 0.0

        # Normalized coherence: min / max ratio — 1 = perfectly scale-invariant
        if signs_agree and fast != 0.0:
            abs_vals = (abs(fast), abs(mid), abs(slow))
            coherence = min(abs_vals) / (max(abs_vals) + 1e-12)
        else:
            coherence = 0.0

        # Blended Sharpe across all three scales
        blended_drift = (fast + mid + slow) / 3.0
        sharpe = blended_drift / (state.fwd_return_std + 1e-9)

        # Confidence = regime certainty × non-anomalousness
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Pack signals; store signed_coherence in expected_return for target()
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=coherence * coherent_dir,   # signed coherence [−1, 1]
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        signed_coherence = signals.expected_return   # coherence * coherent_dir
        coherence        = abs(signed_coherence)
        coherent_dir     = math.copysign(1.0, signed_coherence) if coherence > 0.0 else 0.0

        # Scale-incoherent → flat
        if coherent_dir == 0.0:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        inner = abs(math.tanh(signals.momentum))     # |tanh(sharpe)|
        raw   = coherent_dir * math.tanh(_GAIN * inner * coherence) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh RenormMultiscaleStrategy instance."""
    return RenormMultiscaleStrategy()
