"""Spectral Oscillator strategy — damped harmonic oscillator on manifold drift.

Models the forward-return mean deviation as a damped harmonic oscillator:
    x'' + 2ζω x' + ω² x = 0

x tracks the EMA of fwd_return_mean deviation from its long-run mean.
dx is its velocity (first difference). The restoring acceleration
    a = −ω²x − 2ζω·dx
predicts the oscillator's tendency: when the oscillator is below equilibrium
and turning up (a > 0), go long; when above equilibrium and turning down
(a < 0), go short.
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

NAME = "spectral_oscillator"
DESCRIPTION = (
    "Damped harmonic oscillator on manifold drift: trades the restoring acceleration "
    "predicted by x'' + 2ζω x' + ω² x = 0, gated by regime confidence."
)

# Physics parameters
_OMEGA = 0.15        # natural frequency (rad/bar) — ~42-bar pseudo-period
_ZETA = 0.35         # damping ratio in (0, 1) → under-damped oscillation

# Signal scaling
_GAIN = 8.0          # tanh gain on the restoring acceleration signal
_EMA_ALPHA = 0.12    # EMA smoothing for x (deviation tracking)

# Risk gates
_ANOMALY_GATE = 0.6  # anomaly_score above this → flat
_DEADBAND = 0.04     # |weight| below this collapses to 0


class _SymbolState:
    """Per-symbol oscillator state: position x and velocity dx."""

    __slots__ = ("x", "dx", "ema_mean", "initialized")

    def __init__(self) -> None:
        self.x: float = 0.0
        self.dx: float = 0.0
        self.ema_mean: float = 0.0
        self.initialized: bool = False


class SpectralOscillatorStrategy:
    """Damped harmonic oscillator strategy with per-symbol state.

    target_weight = tanh(gain * a / (fwd_return_std + eps)) * confidence

    where a = −ω²·x − 2·ζ·ω·dx  is the restoring acceleration,
    x is the EMA-smoothed deviation of fwd_return_mean from its running mean,
    dx is the first difference of x (velocity),
    confidence = regime_prob * (1 − anomaly_score).
    """

    def __init__(self) -> None:
        self._state: dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._state:
            self._state[symbol] = _SymbolState()
        return self._state[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        s = self._get_state(sym)

        # Update EMA of fwd_return_mean to track the long-run mean
        if not s.initialized:
            s.ema_mean = state.fwd_return_mean
            s.x = 0.0
            s.dx = 0.0
            s.initialized = True
        else:
            prev_x = s.x
            # Deviation from the running EMA mean
            deviation = state.fwd_return_mean - s.ema_mean
            # Update EMA mean (slow track)
            s.ema_mean += _EMA_ALPHA * 0.5 * (state.fwd_return_mean - s.ema_mean)
            # Update oscillator position x via EMA smoothing of deviation
            new_x = (1.0 - _EMA_ALPHA) * s.x + _EMA_ALPHA * deviation
            s.dx = new_x - prev_x
            s.x = new_x

        # Restoring acceleration: a = −ω²x − 2ζω·dx
        accel = -(_OMEGA ** 2) * s.x - 2.0 * _ZETA * _OMEGA * s.dx

        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=accel,                   # restoring acceleration as momentum proxy
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Normalized restoring acceleration
        norm_accel = signals.momentum / (
            max(abs(signals.expected_return), 1e-9) + 1e-9
        )

        # Map to weight via tanh, scaled by confidence
        weight = math.tanh(_GAIN * norm_accel) * signals.confidence

        # Dead-band suppression
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh SpectralOscillatorStrategy instance."""
    return SpectralOscillatorStrategy()
