"""Fokker-Planck probability-current drift strategy.

The Fokker-Planck equation describes how a probability density p(x,t) evolves
under an SDE  dx = μ dt + σ dW:

    ∂p/∂t = −∂(μ p)/∂x + D ∂²p/∂x²    where D = σ²/2

The probability CURRENT (net flux density) is:

    J = μ·p − D·∂p/∂x

States flow in the direction of J — i.e. the expected motion is the drift μ
MINUS the diffusion constant times the spatial gradient of the density. Where
density is falling (∂p/∂x < 0) the diffusion term reinforces the drift; where
density is rising the diffusion opposes it.

Signal approximation (per symbol, causal):
    D      = k_D · fwd_return_std²          (diffusion ∝ variance)
    ∂p/∂x ≈ density_t − density_{t−1}       (discrete density gradient)
    signal  = fwd_return_mean − k_D · fwd_return_std² · Δdensity

Target weight:
    confidence = regime_prob × (1 − anomaly_score)
    w = tanh(GAIN · signal / (fwd_return_std + ε)) × confidence
    flat when regime == ANOMALY_REGIME or anomaly > ANOMALY_GATE
    dead-band: |w| < DEADBAND → 0
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

NAME = "fokker_planck_drift"
DESCRIPTION = (
    "Fokker-Planck probability-current strategy: trades the net drift of the "
    "state density by computing J = μ·p − D·∂p/∂x, where D ∝ σ² is the "
    "diffusion coefficient and ∂p/∂x is approximated by the per-symbol change "
    "in local manifold density."
)

_EPS = 1e-9
_GAIN = 3.5           # outer tanh gain
_K_D = 2.0            # diffusion scaling: D = _K_D * fwd_return_std²
_ANOMALY_GATE = 0.6   # anomaly_score above this → flat
_DEADBAND = 0.04      # |weight| below this collapses to 0


class _SymbolState:
    """Causal per-symbol state: tracks the previous density observation."""

    __slots__ = ("prev_density", "initialised")

    def __init__(self) -> None:
        self.prev_density: float = 0.0
        self.initialised: bool = False

    def density_gradient(self, density: float) -> float:
        """Return Δdensity = density_t − density_{t−1} (causal, no lookahead)."""
        if not self.initialised:
            self.prev_density = density
            self.initialised = True
            return 0.0
        grad = density - self.prev_density
        self.prev_density = density
        return grad


class FokkerPlanckDriftStrategy:
    """Probability-current strategy derived from the Fokker-Planck equation."""

    def __init__(self) -> None:
        self._states: dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        sym_state = self._get_state(state.symbol)
        d_density = sym_state.density_gradient(state.density)

        # Diffusion coefficient D = k_D * σ²
        diffusion = _K_D * state.fwd_return_std ** 2

        # Probability-current signal: μ − D · ∂p/∂x
        # Normalise by σ so the tanh sees a dimensionless quantity
        fp_signal = (
            state.fwd_return_mean - diffusion * d_density
        ) / (state.fwd_return_std + _EPS)

        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=fp_signal,
            expected_return=state.fwd_return_mean,
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
    return FokkerPlanckDriftStrategy()
