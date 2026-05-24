"""Calm carry strategy.

Harvest steady small positive returns ONLY when conditions are calm and
favorable: low forward-return dispersion, high manifold density, positive
expected drift, and high regime confidence.  In choppy, high-vol, or
uncertain states the strategy stays flat entirely.

Signal:
    sharpe      = fwd_return_mean / (fwd_return_std + eps)
    base        = tanh(gain * tanh(sharpe))
    calm_gate   ~ 1 when density is high AND fwd_return_std is low
    confidence  = regime_prob * (1 - anomaly_score)
    weight      = base * calm_gate * confidence

Gates:
    flat on ANOMALY_REGIME or anomaly_score > 0.6
    flat when fwd_return_mean <= 0  (only harvest positive drift)
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

NAME = "calm_carry"
DESCRIPTION = (
    "Harvests small positive carry gains exclusively in calm, typical market "
    "states: requires high manifold density, low forward-return dispersion, "
    "positive expected drift, and strong regime confidence; stays flat otherwise."
)

_EPS = 1e-9
# tanh gain applied to the Sharpe proxy — moderate so tanh saturates slowly.
_GAIN = 3.0
# Anomaly score threshold above which we go flat.
_ANOMALY_GATE = 0.6
# Dead-band: positions smaller than this are zeroed to avoid noise trading.
_DEADBAND = 0.04

# Calm gate — density leg: logistic sigmoid centred here.
# Density range in synthetic data is ~0.18..0.77; mid at 0.45 lets ~half pass.
_DENSITY_MID = 0.45
_DENSITY_SCALE = 8.0
_DENSITY_CLAMP = 50.0

# Calm gate — std leg: we want LOW std to be allowed.
# Gate falls off as std grows above mid.
# Median std ~0.018; mid at 0.022 lets calmer half pass.
_STD_MID = 0.022   # typical moderate dispersion threshold
_STD_SCALE = 120.0  # steepness


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


def _std_gate(fwd_return_std: float) -> float:
    """Smooth gate in [0, 1]: high std -> ~0, low std -> ~1."""
    s = max(0.0, fwd_return_std)
    return 1.0 / (1.0 + math.exp(_STD_SCALE * (s - _STD_MID)))


class CalmCarryStrategy:
    """Take carry exposure only in calm, high-confidence, low-dispersion states."""

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        calm_gate = _density_gate(state.density) * _std_gate(state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),       # tanh(sharpe) stored for target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * calm_gate, # embed calm_gate into confidence
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Only harvest positive drift (carry mode, not mean-reversion).
        if signals.expected_return <= 0.0:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Reconstruct base signal: tanh(gain * tanh(sharpe)).
        # signals.momentum = tanh(sharpe), so:
        base = math.tanh(_GAIN * signals.momentum)

        # weight = base * calm_gate_fused_confidence
        weight = base * signals.confidence

        # Dead-band.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return CalmCarryStrategy()
