"""Potential-well mean-reversion strategy.

Models the manifold drift as a particle in a quadratic potential well:

    V(x) = 0.5 * kappa * (x - x0)^2

centred at x0, the per-symbol exponential moving average of fwd_return_mean.
The restoring force F = -dV/dx = -kappa * (x - x0) pulls the particle back
toward equilibrium. Well stiffness kappa is proportional to local density
(high density => well-defined, typical regime => stronger restoring force).

Target weight:
    target_weight = tanh(gain * F / (fwd_return_std + eps)) * confidence

where:
    F          = -kappa * (fwd_return_mean - x0)
    kappa      = density  (normalised to [0, 1] by construction)
    confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > ANOMALY_GATE.
Dead-band: |weight| < DEADBAND -> 0.
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

NAME = "potential_well"
DESCRIPTION = (
    "Classical potential-well mean-reversion: restoring force F=-kappa*(x-x0) "
    "pulls toward the per-symbol EMA equilibrium; stiffness kappa scales with "
    "local density; confidence gated by regime_prob and anomaly_score."
)

_EPS = 1e-9
_GAIN = 4.0          # tanh gain: amplifies normalised restoring force into weight
_ANOMALY_GATE = 0.6  # anomaly_score above this -> flat
_DEADBAND = 0.04     # |weight| below this collapses to 0
_EMA_ALPHA = 0.05    # EMA decay for per-symbol equilibrium x0 (slow, ~20-bar half-life)


class PotentialWellStrategy:
    """Particle-in-a-well mean-reversion with per-symbol EMA equilibrium.

    Per-symbol state:
        _x0[symbol]       -- EMA of fwd_return_mean (the potential-well centre)
        _initialised[sym] -- whether we have seen at least one state
    """

    def __init__(self) -> None:
        self._x0: dict[str, float] = {}
        self._initialised: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Lazily initialise x0 to the first observed fwd_return_mean
        if not self._initialised.get(sym, False):
            self._x0[sym] = state.fwd_return_mean
            self._initialised[sym] = True
        else:
            # EMA update — no lookahead: x0 updates BEFORE we use it
            self._x0[sym] = (
                _EMA_ALPHA * state.fwd_return_mean
                + (1.0 - _EMA_ALPHA) * self._x0[sym]
            )

        x0 = self._x0[sym]
        x = state.fwd_return_mean

        # Stiffness kappa proportional to density (typical state => stiffer well)
        kappa = max(0.0, state.density)

        # Restoring force: F = -kappa * (x - x0)
        restoring_force = -kappa * (x - x0)

        # Normalise by volatility to get a dimensionless signal
        normalised_force = restoring_force / (state.fwd_return_std + _EPS)

        # Confidence: high regime probability and low anomaly
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=normalised_force,   # carries the restoring-force signal
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Proposed weight: tanh compresses the restoring force into [-1, 1]
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress micro positions
        if abs(raw) < _DEADBAND:
            raw = 0.0

        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh PotentialWellStrategy instance."""
    return PotentialWellStrategy()
