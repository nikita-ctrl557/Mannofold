"""Canonical partition function strategy — statistical mechanics perspective.

Models three micro-states {long: +1, flat: 0, short: -1} with energies:

    E(+1) = -fwd_return_mean      (low energy = favorable upward drift)
    E( 0) =  lambda_flat          (small flat cost; creates a natural dead-zone)
    E(-1) = +fwd_return_mean      (low energy = favorable downward drift)

Temperature T = fwd_return_std (thermal agitation / uncertainty).

Canonical partition function:
    Z = exp(-E(+1)/(kT)) + exp(-E(0)/(kT)) + exp(-E(-1)/(kT))
      = exp( mu/(kT)) + exp(-lambda/(kT)) + exp(-mu/(kT))

Ensemble-average position (thermodynamically optimal expected exposure):
    <s> = [+1·exp(mu/(kT)) + 0·exp(-lambda/(kT)) + (-1)·exp(-mu/(kT))] / Z
        = [exp(mu/(kT)) - exp(-mu/(kT))] / Z
        = 2·sinh(mu/(kT)) / Z

Target weight:
    weight = tanh(gain · <s>) · confidence
    confidence = regime_prob · (1 - anomaly_score)

Physical interpretation:
  - Low T (low vol): Z dominated by lowest-energy state → decisive signal.
  - High T (high vol): Boltzmann weights equalize → <s> → 0 (cautious).
  - lambda_flat controls the dead-zone width: larger lambda penalises flat
    more, smaller lambda makes flat more competitive vs. long/short.

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

NAME = "partition_function"
DESCRIPTION = (
    "Canonical partition function Z = Σ exp(-E_s/(kT)) over three states "
    "{long:+1, flat:0, short:-1} with E(+1)=-mu, E(0)=lambda, E(-1)=+mu, T=sigma. "
    "Ensemble-average <s> = 2*sinh(mu/kT)/Z is the thermodynamically optimal "
    "expected exposure. weight = tanh(gain*<s>)*confidence, "
    "confidence = regime_prob*(1-anomaly_score). "
    "Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 -> 0."
)

_EPS = 1e-9
_K = 1.0                # Boltzmann constant (dimensionless; absorbed into gain)
_GAIN = 40.0            # tanh amplification on the ensemble average
_LAMBDA_FLAT = 0.001    # flat-state energy cost; controls dead-zone width
_ANOMALY_CUTOFF = 0.6   # hard gate on anomaly score
_DEADBAND = 0.04        # |weight| below this -> flat (avoids churn)


class PartitionFunctionStrategy:
    """Trade according to the canonical ensemble average over long/flat/short states.

    signals():
        - momentum = ensemble average <s> from the canonical partition function.
        - confidence = regime_prob * (1 - anomaly_score).
        - expected_return carries fwd_return_mean for downstream inspection.

    target():
        - Flat when regime_id == ANOMALY_REGIME or anomaly > _ANOMALY_CUTOFF.
        - weight = tanh(gain * <s>) * confidence.
        - Dead-band |weight| < _DEADBAND -> 0.
        - Clamped to [-1, 1].
    """

    def __init__(
        self,
        k: float = _K,
        gain: float = _GAIN,
        lambda_flat: float = _LAMBDA_FLAT,
        anomaly_cutoff: float = _ANOMALY_CUTOFF,
        deadband: float = _DEADBAND,
    ) -> None:
        self._k = k
        self._gain = gain
        self._lambda_flat = lambda_flat
        self._anomaly_cutoff = anomaly_cutoff
        self._deadband = deadband

    def _ensemble_average(self, mu: float, sigma: float) -> float:
        """Compute ensemble-average position <s> from the canonical partition function."""
        kT = self._k * sigma + _EPS
        # Clamp exponent to avoid overflow (exp > ~700 overflows float64)
        exp_arg = min(mu / kT, 500.0)
        z_long  = math.exp( exp_arg)                              # exp(-E(+1)/kT)
        z_short = math.exp(-exp_arg)                              # exp(-E(-1)/kT)
        z_flat  = math.exp(-self._lambda_flat / kT)               # exp(-E(0)/kT)
        Z = z_long + z_flat + z_short
        # <s> = (+1*z_long + 0*z_flat + -1*z_short) / Z
        return (z_long - z_short) / Z

    def signals(self, state: ManifoldState) -> SignalSet:
        avg_s = self._ensemble_average(state.fwd_return_mean, state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=avg_s,
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

        # Amplify ensemble-average position and scale by confidence.
        weight = math.tanh(self._gain * signals.momentum) * signals.confidence

        # Dead-band suppression.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Construct a PartitionFunctionStrategy with default hyperparameters."""
    return PartitionFunctionStrategy()
