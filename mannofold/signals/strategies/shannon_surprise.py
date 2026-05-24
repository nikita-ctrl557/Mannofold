"""Shannon Surprise Strategy.

Information-theoretic strategy using self-information (surprisal) of the
current return move under the recent per-symbol distribution.

Surprisal of an observation x under Gaussian N(μ, σ²):
  I(x) = -log p(x) ∝ 0.5 * z²   where z = (x - μ) / (σ + ε)

LOW surprise  (z ≈ 0, move is expected/predictable)
  → model is well-calibrated here → trade the drift with confidence.

HIGH surprise (large |z|, unpredictable shock)
  → step back, go flat.

target_weight = tanh(gain * tanh(sharpe)) * exp(-0.5 * z² / τ) * confidence

  - tanh(gain * tanh(sharpe)) : direction + drift magnitude
  - exp(-0.5 * z² / τ)        : Gaussian low-surprise gate (τ = temperature)
  - confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > ~0.6.
Dead-band |w| < 0.04 → 0.
Per-symbol state, no lookahead.
"""

from __future__ import annotations

import math
from typing import Dict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "shannon_surprise"
DESCRIPTION = (
    "Information-theoretic strategy using Gaussian surprisal of the current "
    "forward-return mean under a per-symbol running distribution. Low surprisal "
    "→ model is well-calibrated → trade the drift; high surprisal → go flat. "
    "Weight = tanh(gain*tanh(sharpe)) * exp(-0.5*z²/τ) * confidence."
)

_EPS             = 1e-9
_GAIN            = 2.5   # scales tanh(sharpe) → momentum
_TAU             = 1.5   # surprisal temperature (higher = wider acceptance)
_ANOMALY_THRESH  = 0.6
_DEAD_BAND       = 0.04

# EMA decay for running mean/std of fwd_return_mean
_MU_ALPHA        = 0.05
_VAR_ALPHA        = 0.05
_VAR_INIT         = 1e-6  # seed variance (tiny but non-zero)


class _SymbolState:
    """Online EMA tracker for mean and variance of fwd_return_mean."""

    __slots__ = ("mu", "var", "n")

    def __init__(self, mu0: float, var0: float) -> None:
        self.mu  = mu0
        self.var = var0
        self.n   = 1

    def surprisal_and_update(self, x: float) -> float:
        """Return 0.5·z² (proportional to surprisal), then update running stats.

        The update happens AFTER computing z so there is no lookahead.
        """
        z = (x - self.mu) / (math.sqrt(max(self.var, _EPS)) + _EPS)
        surprisal = 0.5 * z * z

        # EMA update (no lookahead)
        self.mu  = _MU_ALPHA  * x          + (1.0 - _MU_ALPHA)  * self.mu
        delta    = x - self.mu
        self.var = _VAR_ALPHA * delta * delta + (1.0 - _VAR_ALPHA) * self.var
        self.n  += 1

        return surprisal


class ShannonSurprise:
    """Per-symbol Shannon-surprisal gated strategy."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str, mu0: float) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(mu0, _VAR_INIT)
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu    = state.fwd_return_mean
        std   = max(state.fwd_return_std, _EPS)

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            return SignalSet(
                ts=state.ts, symbol=state.symbol,
                momentum=0.0, expected_return=mu,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id, confidence=0.0,
            )

        sym = self._get_state(state.symbol, mu)
        surprisal = sym.surprisal_and_update(mu)

        # Sharpe-like signal: drift / local std
        sharpe = mu / (std + _EPS)
        # Low-surprise gate: exp(-surprisal / τ)
        gate = math.exp(-surprisal / max(_TAU, _EPS))
        # Direction+magnitude: tanh(gain * tanh(sharpe))
        direction = math.tanh(_GAIN * math.tanh(sharpe))
        # Combined momentum signal
        momentum = direction * gate

        return SignalSet(
            ts=state.ts, symbol=state.symbol,
            momentum=momentum,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = signals.momentum * signals.confidence

        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh ShannonSurprise strategy instance."""
    return ShannonSurprise()
