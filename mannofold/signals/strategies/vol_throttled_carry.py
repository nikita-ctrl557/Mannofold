"""Vol-throttled carry strategy.

Inverse-variance carry with a dynamic risk throttle based on a per-symbol
EMA of realised fwd_return_std.  When neighbourhood dispersion is elevated
relative to its own smoothed norm, exposure is cut proportionally — giving
higher return-per-unit-risk without abandoning carry altogether.

Signal:  kelly = fwd_return_mean / (fwd_return_std**2 + eps)   [clamped]
         base  = tanh(gain * kelly)
Throttle: ema_std tracked per symbol (EMA of fwd_return_std)
          risk_factor = clamp(target_std / (ema_std + eps), 0, 1)
Weight:   base * risk_factor * confidence
          confidence = regime_prob * (1 - anomaly_score)
Gates:    flat on ANOMALY_REGIME or anomaly_score > 0.6
          dead-band |w| < 0.04 -> 0
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

NAME = "vol_throttled_carry"
DESCRIPTION = (
    "Inverse-variance carry with EMA-based volatility throttle: "
    "exposure is scaled down when neighbourhood dispersion is elevated "
    "relative to its own smoothed norm, controlled by confidence gating."
)

_EPS = 1e-9
# Hard clamp on kelly ratio before tanh to prevent saturation.
_KELLY_CAP = 5.0
# tanh gain applied to clamped kelly.
_GAIN = 2.5
# Anomaly score threshold above which strategy goes flat.
_ANOMALY_GATE = 0.6
# Dead-band: positions smaller than this are zeroed to avoid noise trading.
_DEADBAND = 0.04
# EMA alpha for per-symbol fwd_return_std tracking (higher = faster adaptation).
_EMA_ALPHA = 0.15
# Target std level: risk_factor = clamp(target_std / ema_std, 0, 1).
# Set to a typical moderate std; when ema_std < target_std we cap at 1.0.
_TARGET_STD = 0.02


class VolThrottledCarryStrategy:
    def __init__(
        self,
        gain: float = _GAIN,
        kelly_cap: float = _KELLY_CAP,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
        target_std: float = _TARGET_STD,
    ):
        self._gain = gain
        self._kelly_cap = kelly_cap
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        self._target_std = target_std
        # Per-symbol EMA of fwd_return_std (initialised lazily on first tick).
        self._ema_std: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # --- Update per-symbol EMA of fwd_return_std (no lookahead) ---
        prev_ema = self._ema_std.get(sym, state.fwd_return_std)
        new_ema = self._ema_alpha * state.fwd_return_std + (1.0 - self._ema_alpha) * prev_ema
        self._ema_std[sym] = new_ema

        # --- Inverse-variance / Kelly sizing ---
        variance = state.fwd_return_std ** 2 + _EPS
        kelly = state.fwd_return_mean / variance
        kelly = max(-self._kelly_cap, min(self._kelly_cap, kelly))
        base = math.tanh(self._gain * kelly)

        # --- Risk throttle: cut exposure when dispersion is elevated ---
        risk_factor = min(1.0, max(0.0, self._target_std / (new_ema + _EPS)))

        # --- Confidence: regime stability attenuated by anomaly ---
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Store throttled base in momentum field for use in target().
        throttled = base * risk_factor

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=throttled,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trading near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return VolThrottledCarryStrategy()
