"""Thermal-annealing strategy — statistical mechanics perspective.

Models market volatility as temperature T = fwd_return_std. At high T (hot,
volatile) the system is in a disordered, exploratory state: signals are
unreliable, so we take only small tentative positions. As T cools the market
anneals into a low-energy ordered state and we commit with full conviction.

The Boltzmann-like acceptance factor is:

    conviction_scale = exp(-T / T0)      (1 as T→0, 0 as T→∞)

Target weight formula:

    sharpe    = fwd_return_mean / (fwd_return_std + eps)
    confidence = regime_prob * (1 - anomaly_score)
    weight    = tanh(gain * tanh(sharpe)) * clamp(exp(-T/T0), 0, 1) * confidence

Flat on ANOMALY_REGIME or anomaly > 0.6. Dead-band |w| < 0.04 → 0.
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

NAME = "thermal_annealing"
DESCRIPTION = (
    "Temperature-scaled conviction via Boltzmann acceptance: "
    "conviction_scale = exp(-T/T0) where T = fwd_return_std. "
    "Hot/volatile markets → small exploratory positions; "
    "cool/settled markets → full conviction. "
    "weight = tanh(gain*tanh(sharpe)) * exp(-T/T0) * confidence. "
    "Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 → 0."
)

_EPS = 1e-9
_T0 = 0.015            # reference temperature: ~1.5% daily vol normalises conviction scale
_GAIN = 50.0           # tanh amplification on tanh(sharpe) — sharpens decision boundary
_ANOMALY_CUTOFF = 0.6  # hard gate on anomaly score
_DEADBAND = 0.04       # |weight| below this → flat (avoids churn)


class ThermalAnnealingStrategy:
    """Trade with Boltzmann-scaled conviction: commit only when markets are cool.

    signals():
        - momentum = tanh(sharpe) encodes normalised drift direction.
        - confidence = regime_prob * (1 - anomaly_score).
        - expected_return carries fwd_return_mean for downstream inspection.

    target():
        - Flat when regime_id == ANOMALY_REGIME or anomaly > _ANOMALY_CUTOFF.
        - conviction_scale = exp(-T / T0), clamped to [0, 1].
        - weight = tanh(gain * momentum) * conviction_scale * confidence.
        - Dead-band |weight| < _DEADBAND → 0.
        - Clamped to [-1, 1].
    """

    def __init__(
        self,
        t0: float = _T0,
        gain: float = _GAIN,
        anomaly_cutoff: float = _ANOMALY_CUTOFF,
        deadband: float = _DEADBAND,
    ) -> None:
        self._t0 = t0
        self._gain = gain
        self._anomaly_cutoff = anomaly_cutoff
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
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
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score → flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_cutoff:
            return flat

        # Boltzmann conviction scale: cool markets → scale near 1; hot → near 0.
        # We recover T (fwd_return_std) from expected_return and momentum:
        # momentum = tanh(mean / (std + eps))  ⟹  atanh(momentum) ≈ mean / std
        # So std ≈ mean / atanh(momentum) when momentum != 0.
        # To avoid this fragile inversion, pass T via the signals expected_return
        # interpretation: we embed T into confidence's complement instead.
        # Since signals.momentum = tanh(sharpe) and expected_return = fwd_return_mean,
        # recover std: T = mean / atanh(clamp(momentum)) when |momentum| > eps,
        # else T = _T0 (neutral → half conviction).
        mom = signals.momentum
        mu = signals.expected_return
        if abs(mom) > _EPS:
            mom_clamped = max(-1.0 + _EPS, min(1.0 - _EPS, mom))
            sharpe_est = math.atanh(mom_clamped)
            T = abs(mu) / (abs(sharpe_est) + _EPS) if abs(sharpe_est) > _EPS else self._t0
        else:
            T = self._t0

        conviction_scale = min(1.0, max(0.0, math.exp(-T / self._t0)))

        weight = math.tanh(self._gain * mom) * conviction_scale * signals.confidence

        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Construct a ThermalAnnealingStrategy with default hyperparameters."""
    return ThermalAnnealingStrategy()
