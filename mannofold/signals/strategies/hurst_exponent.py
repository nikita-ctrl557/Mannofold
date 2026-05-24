"""Hurst-exponent strategy: fractal time-series persistence detection.

Estimates a variance-ratio proxy for the Hurst exponent H from a rolling
buffer of fwd_return_mean values:

    H ≈ 0.5 * log(Var(lag k) / Var(lag 1)) / log(k)

  H > 0.5  →  persistent / trending  →  FOLLOW the drift  (+)
  H < 0.5  →  anti-persistent / mean-reverting  →  FADE the drift  (−)
  H ≈ 0.5  →  random walk  →  flat

target_weight = (2H−1 direction) · tanh(gain · tanh(sharpe)) · confidence
where confidence = regime_prob * (1 − anomaly_score).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "hurst_exponent"
DESCRIPTION = (
    "Variance-ratio Hurst exponent proxy: follow drift when H>0.5 (persistent), "
    "fade when H<0.5 (anti-persistent), flat near random-walk H≈0.5."
)

# ── Tunable parameters ────────────────────────────────────────────────────────
_BUFFER_SIZE = 64     # rolling window length for the Hurst buffer
_LAG_K = 8            # coarse lag for variance-ratio (must be < BUFFER_SIZE/2)
_MIN_BARS = 16        # minimum valid buffer length before producing a signal
_GAIN = 2.5           # amplifier inside outer tanh
_ANOMALY_THRESH = 0.6 # anomaly_score above this → flat
_DEAD_BAND = 0.04     # collapse |w| < this to 0
_EPS = 1e-9


def _variance_at_lag(buf: list[float], lag: int) -> float:
    """Non-overlapping block variance at a given lag."""
    n = len(buf)
    n_blocks = n // lag
    if n_blocks < 2:
        return 0.0
    block_means: list[float] = []
    for i in range(n_blocks):
        chunk = buf[i * lag: (i + 1) * lag]
        block_means.append(sum(chunk) / lag)
    grand_mean = sum(block_means) / n_blocks
    var = sum((m - grand_mean) ** 2 for m in block_means) / (n_blocks - 1)
    return var


def _estimate_hurst(buf: list[float], lag_k: int) -> float:
    """Variance-ratio Hurst proxy; returns 0.5 on degenerate inputs."""
    var1 = _variance_at_lag(buf, 1)
    vark = _variance_at_lag(buf, lag_k)
    if var1 < _EPS or vark < _EPS or lag_k <= 1:
        return 0.5
    ratio = vark / var1
    if ratio <= 0.0:
        return 0.5
    h = 0.5 * math.log(ratio) / math.log(lag_k)
    # Clamp to a sensible range; typical values [0.1, 0.9]
    return max(0.05, min(0.95, h))


class HurstExponentStrategy:
    """Per-symbol rolling buffer strategy using the Hurst exponent."""

    def __init__(self) -> None:
        self._buffers: dict[str, Deque[float]] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Initialise buffer on first sight
        if sym not in self._buffers:
            self._buffers[sym] = deque(maxlen=_BUFFER_SIZE)

        buf = self._buffers[sym]
        buf.append(state.fwd_return_mean)

        # Confidence: fuse regime certainty and anomaly suppression
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Not enough data yet → neutral momentum, normal confidence
        if len(buf) < _MIN_BARS:
            return SignalSet(
                ts=state.ts,
                symbol=sym,
                momentum=0.0,
                expected_return=state.fwd_return_mean,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=confidence,
            )

        # Estimate Hurst exponent
        h = _estimate_hurst(list(buf), _LAG_K)

        # direction_scale: +1 when fully persistent, −1 when fully anti-persistent,
        # 0 at exactly H=0.5.  Linear mapping: (2H − 1) ∈ (−1, +1).
        direction_scale = 2.0 * h - 1.0

        # Sharpe-like momentum signal from current bar
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        raw_momentum = direction_scale * math.tanh(sharpe)

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=raw_momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band suppression
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh HurstExponentStrategy instance."""
    return HurstExponentStrategy()
