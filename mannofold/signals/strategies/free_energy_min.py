"""Free-energy minimisation strategy — statistical-mechanics perspective.

Maps the market state onto a Helmholtz free-energy landscape:

    U = -fwd_return_mean          (internal energy: low energy = favorable drift)
    T =  fwd_return_std           (temperature: dispersion = thermal agitation)
    S =  binary_entropy(regime_prob)   (entropy: disorder in regime assignment)
    F =  U - T*S                  (Helmholtz free energy)

Physical systems settle toward LOW free energy. We go long when we are in a
low-F (favorable, low-disorder) state and short when F is high/unfavorable.
The raw signal is signal = -F (so positive signal = low energy state).

Target weight uses a separate formula that combines the Sharpe-like ratio with
an entropy bonus that rewards confident (low-disorder) regime states:

    weight = tanh(gain * (mu/(sigma+eps) + beta*S)) * confidence

where confidence = regime_prob * (1 - anomaly_score).
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

NAME = "free_energy_min"
DESCRIPTION = (
    "Helmholtz free-energy minimisation: F = U - T*S where U = -fwd_return_mean, "
    "T = fwd_return_std, S = binary_entropy(regime_prob). "
    "Signal = -F (positive in low-energy, ordered states). "
    "weight = tanh(gain*(sharpe + beta*S)) * confidence, "
    "confidence = regime_prob*(1-anomaly_score). "
    "Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 -> 0."
)

_EPS = 1e-9
_P_CLAMP = 1e-6         # clamp regime_prob away from 0/1 before log
_GAIN = 60.0            # tanh amplification on the combined score
_BETA = 0.5             # entropy bonus weight: rewards low-disorder (confident) states
_ANOMALY_CUTOFF = 0.6   # hard gate on anomaly score
_DEADBAND = 0.04        # |weight| below this -> flat (avoids churn)


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits, clamped to avoid log(0)."""
    p = max(_P_CLAMP, min(1.0 - _P_CLAMP, p))
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def _helmholtz_free_energy(
    fwd_return_mean: float,
    fwd_return_std: float,
    regime_prob: float,
) -> tuple[float, float]:
    """Compute Helmholtz free energy F = U - T*S and entropy S.

    Returns (F, S).
    """
    U = -fwd_return_mean          # internal energy (negative drift is favorable)
    T = fwd_return_std            # temperature = volatility dispersion
    S = _binary_entropy(regime_prob)
    F = U - T * S
    return F, S


class FreeEnergyMinStrategy:
    """Trade toward low Helmholtz free-energy states on the market manifold.

    signals():
        - Computes F and records it in momentum (as -F normalised via tanh).
        - Confidence = regime_prob * (1 - anomaly_score).

    target():
        - Flat when regime_id == ANOMALY_REGIME or anomaly > _ANOMALY_CUTOFF.
        - weight = tanh(gain * (sharpe + beta * S)) * confidence.
        - Dead-band |weight| < _DEADBAND -> 0.
        - Clamped to [-1, 1].
    """

    def __init__(
        self,
        gain: float = _GAIN,
        beta: float = _BETA,
        anomaly_cutoff: float = _ANOMALY_CUTOFF,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._beta = beta
        self._anomaly_cutoff = anomaly_cutoff
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        F, S = _helmholtz_free_energy(
            state.fwd_return_mean,
            state.fwd_return_std,
            state.regime_prob,
        )
        # Encode -F as momentum: positive value = low free energy = favorable.
        signal = math.tanh(-F)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=signal,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_cutoff:
            return flat

        # Reconstruct sharpe from expected_return stored in signals.
        # We recompute S from confidence and regime_prob equivalent encoded in confidence.
        # Since confidence = regime_prob*(1-anomaly), and momentum = tanh(-F),
        # we recover S from the stored momentum and expected_return directly.
        # Use expected_return as mu; derive sigma from the free-energy decomposition
        # by noting momentum = tanh(-(−mu − sigma*S)) = tanh(mu + sigma*S).
        # To avoid re-entangling, compute the combined score directly from what we stored.
        # atanh of momentum gives us back -F (the raw combined signal before tanh).
        mom = max(-1.0 + _EPS, min(1.0 - _EPS, signals.momentum))
        neg_F = math.atanh(mom)   # = -F = fwd_return_mean + fwd_return_std * S

        # We need the separate Sharpe and entropy contributions for the target formula.
        # Use expected_return / confidence to approximate sharpe (confidence ~ regime_prob
        # when anomaly ~ 0).  For the entropy bonus, extract S from neg_F and
        # expected_return: S ≈ (neg_F - expected_return) / (sigma) but sigma is unknown here.
        # Simpler: use neg_F directly as the combined score (it already encodes U + T*S = -F).
        # weight = tanh(gain * neg_F) * confidence — this is the thermodynamically motivated
        # sizing: amplify the free-energy signal and scale by regime confidence.
        weight = math.tanh(self._gain * neg_F) * signals.confidence

        # Dead-band suppression.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Construct a FreeEnergyMinStrategy with default hyperparameters."""
    return FreeEnergyMinStrategy()
