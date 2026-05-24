"""Confidence-scaled Kelly strategy: fractional-Kelly inverse-variance sizing scaled by confidence squared.

Team RISK-MANAGED-CARRY member strategy. Concentrates risk into high-confidence states by squaring
the confidence factor (regime_prob * (1 - anomaly_score)), so the engine only bets big when
both regime probability is high AND anomaly score is low. Goes flat in anomalous regimes or
when anomaly_score exceeds the gate threshold.

Kelly criterion: f* = mu / sigma^2
Base signal: tanh(gain * 0.5 * kelly_clamped)
Final weight: base * (confidence**2) -> dead-band -> clamp
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

NAME = "confidence_kelly"
DESCRIPTION = (
    "Confidence-squared fractional-Kelly inverse-variance sizing: "
    "target_weight = tanh(gain * 0.5 * kelly_clamped) * confidence^2, "
    "where confidence = regime_prob * (1 - anomaly_score). "
    "Concentrates risk into high-confidence states; flat in anomalous regimes."
)

_EPS = 1e-9
# Hard cap on |0.5 * kelly| before tanh squash.
_KELLY_CAP = 3.0
# tanh gain applied to half-Kelly value.
_GAIN = 2.0
# Anomaly gate: go flat if anomaly_score exceeds this.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions to avoid noise trading.
_DEADBAND = 0.04


class ConfidenceKellyStrategy:
    """Fractional-Kelly sizing scaled by confidence squared for risk concentration."""

    def signals(self, state: ManifoldState) -> SignalSet:
        # Half-Kelly inverse-variance ratio: mu / sigma^2, with 0.5 fraction.
        variance = state.fwd_return_std ** 2 + _EPS
        kelly_raw = state.fwd_return_mean / variance
        kelly_half = 0.5 * kelly_raw
        kelly_clamped = max(-_KELLY_CAP, min(_KELLY_CAP, kelly_half))

        # Base signal: tanh squash of half-Kelly with gain.
        base = math.tanh(_GAIN * kelly_clamped)

        # Confidence: regime stability attenuated by anomaly proximity.
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Embed confidence-squared scaling into momentum for downstream use.
        momentum = base * (confidence ** 2)

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

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Weight already incorporates confidence^2 via momentum.
        weight = signals.momentum

        # Dead-band: collapse negligible weights to flat to avoid noise trades.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return ConfidenceKellyStrategy()
