"""Anomaly-fade-recovery strategy: enter as dislocations resolve back toward normal.

Tracks per-symbol anomaly history to detect RESOLVING anomalies: when anomaly_score
has dropped sharply from a recent peak back toward normal levels, it signals a
recovery entry in the drift direction. When anomaly is rising or flat-high, stay flat.
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

NAME = "anomaly_fade_recovery"
DESCRIPTION = (
    "Entry on resolving anomalies: size into drift direction as anomaly_score "
    "drops from a recent peak back toward normal; flat when rising or anomalous regime."
)

# Tunable knobs
_GAIN = 2.5         # amplifier inside double-tanh: tanh(gain * tanh(sharpe))
_DEAD_BAND = 0.04   # collapse |weight| below this to 0
_PEAK_WINDOW = 8    # how many bars to remember for rolling peak anomaly
_RECOVERY_MIN = 0.15  # minimum recede fraction (peak - current) / peak to register recovery
_ANOMALY_ENTRY_CAP = 0.85  # if current anomaly is still above this, stay flat regardless


class AnomalyFadeRecoveryStrategy:
    """Take positions as anomaly dislocations resolve back toward the manifold.

    Mechanic:
    - Maintain a rolling window of recent anomaly_scores per symbol.
    - recovery_factor = clamp((peak - current) / (peak + eps), 0, 1)
      where peak is the maximum anomaly_score over the last _PEAK_WINDOW bars.
    - When anomaly was elevated and is now dropping, recovery_factor is large.
    - When anomaly is rising or still near its peak, recovery_factor is small/zero.
    - weight = sign(fwd_return_mean) * tanh(gain * tanh(sharpe)) * recovery_factor * confidence
    """

    def __init__(self) -> None:
        # per-symbol rolling window of recent anomaly scores
        self._anomaly_history: dict[str, list[float]] = {}

    def _update_and_get_recovery(self, symbol: str, current_anomaly: float) -> float:
        """Update rolling window and return recovery_factor in [0, 1]."""
        history = self._anomaly_history.setdefault(symbol, [])
        history.append(current_anomaly)
        if len(history) > _PEAK_WINDOW:
            history.pop(0)

        if len(history) < 2:
            return 0.0

        # Peak is the maximum anomaly seen over the window (excluding current)
        peak = max(history[:-1])

        if peak < _RECOVERY_MIN:
            # Was never meaningfully elevated — no recovery trade
            return 0.0

        # How much has anomaly receded from the peak?
        recede = (peak - current_anomaly) / (peak + 1e-9)
        recovery_factor = max(0.0, min(1.0, recede))

        # Only trade if recede exceeds minimum threshold
        if recovery_factor < _RECOVERY_MIN:
            recovery_factor = 0.0

        return recovery_factor

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
        momentum = math.tanh(_GAIN * math.tanh(sharpe))
        confidence = max(0.0, min(1.0, state.regime_prob))

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
        # Hard flat on anomalous regime sentinel
        if signals.regime_id == ANOMALY_REGIME:
            # Still update history so we track the anomaly trajectory
            self._update_and_get_recovery(signals.symbol, signals.anomaly)
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Flat if anomaly is still too high (not recovered enough)
        if signals.anomaly > _ANOMALY_ENTRY_CAP:
            self._update_and_get_recovery(signals.symbol, signals.anomaly)
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        recovery_factor = self._update_and_get_recovery(signals.symbol, signals.anomaly)

        # weight = direction * magnitude * recovery_factor * confidence
        weight = signals.momentum * recovery_factor * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh AnomalyFadeRecoveryStrategy instance."""
    return AnomalyFadeRecoveryStrategy()
