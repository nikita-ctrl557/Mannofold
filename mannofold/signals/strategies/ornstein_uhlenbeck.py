"""Ornstein-Uhlenbeck mean-reversion strategy.

Models the neighbourhood drift as an OU process:
    dx = θ(μ − x)dt + σdW

where x = fwd_return_mean from the manifold neighbourhood. Maintains a per-symbol
EMA estimate of the long-run mean μ and estimates the mean-reversion speed θ from
the autocorrelation/decay of recent deviations. Bets on reversion toward μ, scaled
by reversion speed and regime confidence.
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

NAME = "ornstein_uhlenbeck"
DESCRIPTION = (
    "OU mean-reversion: bets on drift reverting toward its long-run mean μ, "
    "scaled by estimated reversion speed θ and regime confidence."
)

_EPS = 1e-9
_GAIN = 4.0           # outer tanh gain on the OU signal
_ANOMALY_GATE = 0.6   # anomaly_score above this -> go flat
_DEADBAND = 0.04      # |weight| below this collapses to 0
_EMA_ALPHA = 0.05     # EMA smoothing for μ (long-run mean estimate)
_WINDOW = 20          # rolling window for θ estimation via lag-1 autocorrelation


class _SymbolState:
    """Mutable per-symbol OU parameter estimates."""

    __slots__ = ("mu", "deviations", "initialised")

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.deviations: deque[float] = deque(maxlen=_WINDOW)
        self.initialised: bool = False

    def update(self, x: float) -> None:
        if not self.initialised:
            self.mu = x
            self.initialised = True
        else:
            self.mu = (1.0 - _EMA_ALPHA) * self.mu + _EMA_ALPHA * x
        self.deviations.append(x - self.mu)

    def theta(self) -> float:
        """Estimate θ from lag-1 autocorrelation of deviations.

        For a discrete-time OU process sampled at Δt=1:
            e^{-θΔt} ≈ corr(ε_t, ε_{t-1})
        so θ ≈ -ln(max(corr, ε)).
        """
        devs = list(self.deviations)
        n = len(devs)
        if n < 4:
            return 1.0  # default moderate reversion speed
        mean_d = sum(devs) / n
        d = [v - mean_d for v in devs]
        var = sum(v * v for v in d) / n
        if var < _EPS:
            return 1.0
        cov = sum(d[i] * d[i - 1] for i in range(1, n)) / (n - 1)
        rho = max(cov / var, _EPS)          # clamp: only positive autocorr -> OU
        rho = min(rho, 1.0 - _EPS)          # keep ln well-defined
        return -math.log(rho)               # θ = -ln(ρ), ρ=e^{-θ}


class OrnsteinUhlenbeckStrategy:
    """OU drift mean-reversion strategy with per-symbol parameter tracking."""

    def __init__(self) -> None:
        self._states: dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        sym_state = self._get_state(state.symbol)
        sym_state.update(state.fwd_return_mean)

        mu = sym_state.mu
        theta = sym_state.theta()
        deviation = mu - state.fwd_return_mean   # direction of reversion
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # OU signal: θ * (μ − x) / σ — positive -> expect upward reversion
        ou_signal = theta * deviation / (state.fwd_return_std + _EPS)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=ou_signal,
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
    return OrnsteinUhlenbeckStrategy()
