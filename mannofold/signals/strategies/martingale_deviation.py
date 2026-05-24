"""Martingale-deviation strategy: test the efficient-market martingale null hypothesis.

The efficient-market null is a MARTINGALE — zero expected drift. This strategy
only trades when the neighbourhood forward-return mean is STATISTICALLY
SIGNIFICANTLY different from zero, as measured by a t-statistic:

    t = fwd_return_mean / (fwd_return_std / sqrt(n_eff) + eps)

where n_eff is a proxy for the effective neighbourhood sample size derived from
the local manifold density (scaled around a baseline of ~25 observations).

Below the critical t-threshold the drift is indistinguishable from zero under
the martingale null — remain flat. Above the threshold trade the drift, with
size proportional to how far the t-stat exceeds the critical value.

    target_weight = sign(mean) * tanh(gain * max(0, |t| - t_crit)) * confidence
    confidence    = regime_prob * (1 - anomaly_score)

Flat when: ANOMALY_REGIME, anomaly_score > 0.6, |t| <= t_crit, or |weight| < 0.04.
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

NAME = "martingale_deviation"
DESCRIPTION = (
    "Trade only when neighbourhood drift is statistically significantly non-zero "
    "(rejects the martingale / efficient-market null via a t-test); "
    "flat on ANOMALY_REGIME, high anomaly, or below critical t-threshold."
)

# --------------------------------------------------------------------------- #
# Tunable parameters
# --------------------------------------------------------------------------- #
_EPS = 1e-9
_T_CRIT = 2.0          # two-tailed t-critical (~95 % at n~25)
_GAIN = 1.5            # amplifier: controls how fast size ramps above t_crit
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat
_DEAD_BAND = 0.04      # collapse |weight| below this to flat
# Effective-n parameters: n_eff = N_BASE + DENSITY_SCALE * density
# density is a local neighbourhood count proxy; typical range 0 .. ~5+
_N_BASE = 15.0         # minimum n_eff contribution (constant floor)
_N_DENSITY_SCALE = 6.0 # extra observations credited per unit of density


class MartingaleDeviationStrategy:
    """Reject the martingale null only when drift is statistically significant."""

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        # Effective sample size proxy from local density
        n_eff = _N_BASE + _N_DENSITY_SCALE * max(0.0, state.density)

        # t-statistic: drift / SE, where SE = std / sqrt(n_eff)
        se = state.fwd_return_std / (math.sqrt(max(1.0, n_eff)) + _EPS)
        t_stat = state.fwd_return_mean / (se + _EPS)

        # Confidence: regime stability * (1 - anomaly)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=t_stat,          # repurpose momentum slot to carry t-stat
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard guards: anomalous regime or high anomaly -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        t_stat = signals.momentum  # stored in momentum field from signals()
        abs_t = abs(t_stat)

        # Martingale null: drift indistinguishable from zero -> flat
        if abs_t <= _T_CRIT:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Excess significance above critical threshold
        excess = abs_t - _T_CRIT

        # Weight proportional to excess significance, bounded by tanh
        direction = 1.0 if t_stat > 0.0 else -1.0
        raw = direction * math.tanh(_GAIN * excess) * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MartingaleDeviationStrategy instance (no arguments required)."""
    return MartingaleDeviationStrategy()
