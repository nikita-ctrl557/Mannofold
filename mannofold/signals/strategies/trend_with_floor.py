"""Trend-with-floor strategy: always-participating trend follower with minimum exposure floor.

Month-over-month consistency: keeps a small always-on participation floor (+0.15) in
non-anomalous states to capture broad drift every month, scaling up with conviction.

weight = clamp(0.15 + tanh(gain * tanh(sharpe)), -0.2, 1.0) * confidence, EMA-smoothed.
confidence = regime_prob * (1 - anomaly_score).

Goes flat ONLY on ANOMALY_REGIME or anomaly_score > ~0.6. Dead-band does NOT zero
the floor — it only prevents near-zero oscillation around the floor boundary.
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

NAME = "trend_with_floor"
DESCRIPTION = (
    "Always-participating long-biased trend follower with a +0.15 minimum exposure floor "
    "in non-anomalous states; scales up with Sharpe conviction; flat only on ANOMALY_REGIME "
    "or anomaly > 0.6. EMA-smoothed for consistent month-over-month participation."
)

_EPS = 1e-9
_GAIN = 3.0           # tanh gain on the Sharpe proxy (moderate — floor carries the base)
_FLOOR = 0.15         # always-on participation floor in normal states
_MAX_SHORT = -0.2     # long-bias: cap short side
_ANOMALY_GATE = 0.6   # anomaly_score above which we go fully flat
_EMA_ALPHA = 0.20     # smoothing alpha: new = alpha*raw + (1-alpha)*prev


class TrendWithFloorStrategy:
    """Trend-follow manifold drift with a persistent participation floor.

    Signal construction:
      sharpe      = fwd_return_mean / (fwd_return_std + eps)
      confidence  = regime_prob * (1 - anomaly_score)
      raw_weight  = clamp(floor + tanh(gain * tanh(sharpe)), max_short, 1.0) * confidence
      target      = EMA(raw_weight, alpha)

    The floor ensures we always hold some long exposure when the market is in a
    known non-anomalous regime, capturing broad upward drift even in low-conviction
    periods.  Conviction on top of the floor scales up to full exposure.
    """

    def __init__(
        self,
        gain: float = _GAIN,
        floor: float = _FLOOR,
        max_short: float = _MAX_SHORT,
        anomaly_gate: float = _ANOMALY_GATE,
        ema_alpha: float = _EMA_ALPHA,
    ):
        self._gain = gain
        self._floor = floor
        self._max_short = max_short
        self._anomaly_gate = anomaly_gate
        self._ema_alpha = ema_alpha
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std
        momentum = math.tanh(self._gain * math.tanh(sharpe))
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
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
        prev_ema = self._ema.get(sym, 0.0)

        # Hard gate: anomalous regime or extreme anomaly score -> fully flat, decay EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            self._ema[sym] = (1.0 - self._ema_alpha) * prev_ema
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Floor + trend: always-on floor plus scaled conviction on top.
        raw = self._floor + signals.momentum  # floor + tanh(gain*tanh(sharpe))
        # Clamp: bounded by max_short on downside, 1.0 on upside.
        raw = max(self._max_short, min(1.0, raw))
        # Scale by confidence (regime stability * anomaly discount).
        raw_weighted = raw * signals.confidence

        # EMA smoothing for consistent month-over-month exposure.
        smoothed = self._ema_alpha * raw_weighted + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Hard clip.
        weight = max(-1.0, min(1.0, smoothed))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return TrendWithFloorStrategy()
