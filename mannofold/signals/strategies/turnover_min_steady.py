"""Turnover-minimising steady strategy: extreme low-turnover for smooth monthly returns.

CONCEPT: wide entry/exit hysteresis (big no-trade zone) + very heavy per-symbol
EMA smoothing (alpha~0.08) so positions only change on strong, sustained signals.
Holds steady through noise => smoother monthly returns, lower friction/costs.

weight = tanh(gain * tanh(sharpe)) * confidence
confidence = regime_prob * (1 - anomaly_score)
smoothed   = alpha * raw + (1 - alpha) * prev_ema   (alpha ~ 0.08)
dead-band  : |smoothed| < 0.04 -> 0
hysteresis : only update position when |smoothed - current_pos| > threshold
Flat on ANOMALY_REGIME or anomaly > 0.6.
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

NAME = "turnover_min_steady"
DESCRIPTION = (
    "Extreme low-turnover strategy for consistent monthly returns: very heavy "
    "per-symbol EMA smoothing (alpha~0.08) and wide entry/exit hysteresis "
    "(big no-trade zone). Position only changes on strong, sustained signals. "
    "weight = tanh(gain*tanh(sharpe)) * confidence, confidence = "
    "regime_prob*(1-anomaly_score). Flat on ANOMALY_REGIME or anomaly>0.6; "
    "dead-band |w|<0.04->0; per-symbol hysteresis threshold=0.07."
)

_EPS = 1e-9
# Moderate gain for conviction without over-trading at low Sharpe.
_GAIN = 3.0
# Hard anomaly gate: go flat above this score.
_ANOMALY_GATE = 0.6
# Wide dead-band: kills small noisy weights; reduces turnover significantly.
_DEADBAND = 0.04
# Extra-heavy smoothing — slower-moving than longhorizon_compounder.
_EMA_ALPHA = 0.08  # new_ema = alpha * raw + (1 - alpha) * prev_ema
# Hysteresis threshold: only change current_pos when smoothed drifts this far.
_HYSTERESIS = 0.07


class TurnoverMinSteady:
    """Extreme low-turnover strategy optimised for month-over-month consistency."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
        hysteresis: float = _HYSTERESIS,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        self._hysteresis = hysteresis
        # Per-symbol EMA and last-committed position — no lookahead.
        self._ema: dict[str, float] = {}
        self._pos: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        std = state.fwd_return_std + _EPS
        sharpe = state.fwd_return_mean / std

        # Double-tanh conviction: inner squash Sharpe, outer applies gain.
        momentum = math.tanh(self._gain * math.tanh(sharpe))

        # Confidence: regime stability discounted by anomaly proximity.
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
        prev_pos = self._pos.get(sym, 0.0)

        # Hard gates: anomalous regime or high anomaly -> go flat, decay EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            decayed = (1.0 - self._ema_alpha) * prev_ema
            self._ema[sym] = decayed
            # Decay committed position toward zero gradually too.
            new_pos = (1.0 - self._ema_alpha) * prev_pos
            new_pos = 0.0 if abs(new_pos) < self._deadband else new_pos
            self._pos[sym] = new_pos
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: conviction scaled by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Extra-heavy EMA smoothing — positions change very slowly.
        smoothed = self._ema_alpha * raw_weight + (1.0 - self._ema_alpha) * prev_ema
        self._ema[sym] = smoothed

        # Wide dead-band: flat near zero to avoid noise trades.
        candidate = 0.0 if abs(smoothed) < self._deadband else smoothed

        # Hysteresis: only commit a new position if it differs enough from current.
        if abs(candidate - prev_pos) > self._hysteresis:
            new_pos = candidate
        else:
            new_pos = prev_pos

        # Hard clip to [-1, 1].
        new_pos = max(-1.0, min(1.0, new_pos))
        self._pos[sym] = new_pos
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=new_pos)


def build() -> Strategy:
    return TurnoverMinSteady()
