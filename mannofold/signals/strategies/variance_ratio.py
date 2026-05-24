"""Lo-MacKinlay Variance-Ratio test strategy.

The variance-ratio VR(q) = Var(q-period sums) / (q * Var(1-period returns))
measures autocorrelation structure:
  VR > 1  =>  positive autocorrelation  =>  trending  =>  FOLLOW drift
  VR < 1  =>  negative autocorrelation  =>  mean-reverting  =>  FADE drift
  VR = 1  =>  random walk (martingale)  =>  no edge

Per-symbol rolling buffer accumulates fwd_return_mean observations, then
computes VR(q). Direction is tanh(k * (VR - 1)), which smoothly transitions
from fade (<1) to follow (>1). Final weight adds a Sharpe-informed magnitude
and regime/anomaly confidence gate.
"""

from __future__ import annotations

import math
from collections import deque

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "variance_ratio"
DESCRIPTION = (
    "Lo-MacKinlay variance-ratio test: VR>1 => positive autocorrelation => "
    "trend-follow; VR<1 => mean-reversion => fade; direction_factor = "
    "tanh(k*(VR-1)) weighted by regime confidence and anomaly gate."
)

# ── Tunable knobs ──────────────────────────────────────────────────────────
_BUFFER_SIZE = 60        # rolling window for fwd_return_mean observations
_Q = 5                   # lag period for VR test (q-period sums)
_K = 4.0                 # sharpness of tanh(k*(VR-1)) direction sigmoid
_GAIN = 3.0              # outer tanh gain for Sharpe-magnitude sizing
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04        # |weight| below this collapses to 0
_EPS = 1e-12             # numerical floor for variance
# ──────────────────────────────────────────────────────────────────────────


def _variance_ratio(buf: list[float], q: int) -> float:
    """Compute VR(q) from a 1-d list of return observations.

    VR(q) = Var(sum of q consecutive returns) / (q * Var(single-step returns))

    Uses unbiased sample variances. Returns 1.0 (random-walk default) if the
    buffer is too short or variance is degenerate.
    """
    n = len(buf)
    if n < max(q + 1, 4):
        return 1.0

    # Variance of single-period returns
    mu1 = sum(buf) / n
    var1 = sum((x - mu1) ** 2 for x in buf) / (n - 1)
    if var1 < _EPS:
        return 1.0

    # Build overlapping q-period sums
    q_sums = [sum(buf[i : i + q]) for i in range(n - q + 1)]
    nq = len(q_sums)
    if nq < 2:
        return 1.0
    mu_q = sum(q_sums) / nq
    var_q = sum((x - mu_q) ** 2 for x in q_sums) / (nq - 1)

    return var_q / (q * var1 + _EPS)


class VarianceRatioStrategy:
    """Per-symbol rolling VR test with trend/fade direction switching."""

    def __init__(self) -> None:
        # Per-symbol circular buffers of fwd_return_mean
        self._buffers: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Maintain per-symbol rolling buffer
        if sym not in self._buffers:
            self._buffers[sym] = deque(maxlen=_BUFFER_SIZE)
        buf = self._buffers[sym]
        buf.append(state.fwd_return_mean)

        # Need at least q+1 observations before computing VR
        if len(buf) < _Q + 1:
            return SignalSet(
                ts=state.ts,
                symbol=sym,
                momentum=0.0,
                expected_return=0.0,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=0.0,
            )

        vr = _variance_ratio(list(buf), _Q)

        # Direction: VR>1 -> follow (positive), VR<1 -> fade (negative)
        direction_factor = math.tanh(_K * (vr - 1.0))

        # Sharpe of neighbourhood drift (magnitude signal)
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # Combined momentum signal: direction * magnitude
        momentum = direction_factor * math.tanh(_GAIN * math.tanh(sharpe))

        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
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

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Weight = direction_factor * magnitude * confidence
        weight = signals.momentum * signals.confidence

        # Dead-band: suppress noisy micro positions
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh VarianceRatioStrategy instance."""
    return VarianceRatioStrategy()
