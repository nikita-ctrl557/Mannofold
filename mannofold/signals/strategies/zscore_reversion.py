"""Z-score mean-reversion strategy variant.

Maintains per-symbol EMA mean and EMA variance of the neighbourhood
fwd_return_mean (past-only, no lookahead).  Computes a standardised
z-score and fades extremes with a tanh, expecting reversion toward the
running mean.  Confidence is gated by regime stability and anomaly score.
Goes flat in anomalous regimes or when the anomaly score is high.
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

NAME = "zscore_reversion"
DESCRIPTION = (
    "Z-score mean-reversion: EMAs track neighbourhood drift; "
    "fades standardised extremes expecting reversion, gated by regime confidence."
)

_EPS = 1e-9
_GAIN = 2.5          # tanh gain applied to the z-score
_ANOMALY_GATE = 0.6  # anomaly_score above this -> go flat
_DEADBAND = 0.04     # |weight| below this collapses to 0
_ALPHA = 0.05        # EMA decay (≈ 2/(N+1) for N≈39 bars)


class ZScoreReversionStrategy:
    """Per-symbol EMA z-score mean-reversion.

    Running update (past-only):
        ema_mean  <- alpha * x  + (1 - alpha) * ema_mean
        ema_var   <- alpha * (x - ema_mean)^2 + (1 - alpha) * ema_var

    Signal:
        z          = (x - ema_mean) / (sqrt(ema_var) + eps)
        raw_weight = -tanh(gain * z)          # short when z high, long when z low
        weight     = raw_weight * confidence  # gate by regime_prob * (1 - anomaly)

    Gates:
        - regime_id == ANOMALY_REGIME   -> flat
        - anomaly_score > _ANOMALY_GATE -> flat
        - |weight| < _DEADBAND          -> flat
        - clamp to [-1, 1]
    """

    def __init__(self) -> None:
        # keyed by symbol
        self._ema_mean: dict[str, float] = {}
        self._ema_var: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        x = state.fwd_return_mean

        # Initialise EMA on first observation
        if sym not in self._ema_mean:
            self._ema_mean[sym] = x
            self._ema_var[sym] = (state.fwd_return_std ** 2) or _EPS

        # Update EMAs (deterministic, past-only)
        prev_mean = self._ema_mean[sym]
        new_mean = _ALPHA * x + (1.0 - _ALPHA) * prev_mean
        new_var = _ALPHA * (x - new_mean) ** 2 + (1.0 - _ALPHA) * self._ema_var[sym]

        self._ema_mean[sym] = new_mean
        self._ema_var[sym] = max(new_var, _EPS)

        z = (x - new_mean) / (math.sqrt(self._ema_var[sym]) + _EPS)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=z,              # re-use momentum slot for z-score
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        z = signals.momentum
        raw_weight = -math.tanh(_GAIN * z)
        weight = raw_weight * signals.confidence

        # Dead-band
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return ZScoreReversionStrategy()
