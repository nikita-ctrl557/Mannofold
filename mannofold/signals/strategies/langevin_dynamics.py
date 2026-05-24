"""Langevin dynamics strategy.

Models the drift as a particle obeying the Langevin equation:

    m·dv/dt = −γv − ∇V(x) + noise

In the trading context:
    x        = fwd_return_mean (the "position" of the drift particle)
    velocity = per-symbol first-difference of x  (dx = x_t − x_{t−1})
    −∇V(x)  = −κ(x − μ)   restoring force toward long-run mean μ
    −γv      = −γ·dx        friction proportional to velocity

The deterministic predicted next move is:

    Δx_pred ≈ −κ(x − μ) − γ·dx

This is traded as the signal.  Friction γ is elevated when anomaly_score is
high, so the strategy damps itself during turbulent regimes.

Target weight:
    confidence  = regime_prob * (1 − anomaly_score)
    target_weight = tanh(gain · Δx_pred / (fwd_return_std + ε)) · confidence

Flat on ANOMALY_REGIME or anomaly_score > ANOMALY_GATE.
Dead-band: |weight| < DEADBAND → 0.
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

NAME = "langevin_dynamics"
DESCRIPTION = (
    "Langevin-equation strategy: treats fwd_return_mean as a particle under "
    "restoring force −κ(x−μ) plus friction −γ·dx; friction increases with "
    "anomaly_score to damp in turbulence; trades the deterministic predicted drift."
)

_EPS = 1e-9
_GAIN = 4.0            # tanh gain on the normalised Langevin signal
_ANOMALY_GATE = 0.6    # anomaly_score above this -> go flat
_DEADBAND = 0.04       # |weight| below this collapses to 0
_EMA_ALPHA = 0.05      # EMA smoothing for μ (long-run mean, ~20-bar half-life)
_KAPPA = 0.3           # base restoring-force stiffness
_GAMMA_BASE = 0.2      # base friction coefficient
_GAMMA_ANOM = 1.5      # additional friction contribution when anomaly → 1


class _SymbolState:
    """Mutable per-symbol Langevin parameter estimates."""

    __slots__ = ("mu", "prev_x", "initialised")

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.prev_x: float | None = None
        self.initialised: bool = False

    def update(self, x: float) -> tuple[float, float]:
        """Update state and return (mu, dx).

        dx is the first-difference (velocity); returns 0.0 on first observation.
        """
        if not self.initialised:
            self.mu = x
            self.initialised = True
            dx = 0.0
        else:
            self.mu = (1.0 - _EMA_ALPHA) * self.mu + _EMA_ALPHA * x
            dx = x - (self.prev_x if self.prev_x is not None else x)
        self.prev_x = x
        return self.mu, dx


class LangevinDynamicsStrategy:
    """Langevin-dynamics drift strategy with per-symbol particle state."""

    def __init__(self) -> None:
        self._states: dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        sym_state = self._get_state(state.symbol)
        x = state.fwd_return_mean
        mu, dx = sym_state.update(x)

        # Friction coefficient rises with anomaly (damp in turbulence)
        gamma = _GAMMA_BASE + _GAMMA_ANOM * state.anomaly_score

        # Deterministic Langevin predicted drift:
        #   Δx_pred = −κ(x − μ) − γ·dx
        restoring = -_KAPPA * (x - mu)
        friction = -gamma * dx
        delta_pred = restoring + friction

        # Confidence: high regime probability and low anomaly
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Normalise signal by volatility for scale-invariance
        normalised = delta_pred / (state.fwd_return_std + _EPS)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=normalised,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = math.tanh(_GAIN * signals.momentum) * signals.confidence

        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh LangevinDynamicsStrategy instance."""
    return LangevinDynamicsStrategy()
