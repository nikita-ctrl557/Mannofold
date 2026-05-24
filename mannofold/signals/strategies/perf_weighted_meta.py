"""Hit-rate-weighted blend strategy: ensemble of drift, carry, and reversion
sub-signals where each sub-signal is weighted by its trailing per-symbol HIT-RATE.

Hit-rate is maintained as an EMA of (1.0 if previous sub-signal direction matched
the subsequently-observed move, else 0.0). Only past information is used — no
lookahead. Sub-signals with hit-rate <= 0.5 (coin-flip or worse) receive zero
weight; only better-than-chance sub-signals contribute to the composite.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "perf_weighted_meta"
DESCRIPTION = (
    "Ensemble of drift, carry, and reversion sub-signals weighted by trailing "
    "per-symbol hit-rate (EMA). Only sub-signals beating coin-flip contribute."
)

_EPS             = 1e-9
_CARRY_CLAMP     = 5.0
_GAIN            = 2.5
_HR_ALPHA        = 0.10    # EMA decay for hit-rate (lower = longer memory)
_ANOMALY_THRESH  = 0.6
_DEAD_BAND       = 0.04
_N_SUBS          = 3       # drift, carry, reversion
_HR_INIT         = 0.5     # start at neutral (coin-flip) so weight starts at 0


class _SymbolState:
    """Per-symbol online learning state tracking trailing hit-rate per sub-signal."""

    def __init__(self) -> None:
        # EMA of hit indicator (1.0 = hit, 0.0 = miss) per sub-signal
        self.hit_rates: List[float] = [_HR_INIT] * _N_SUBS
        # Previous sub-signal values (signed), to compare next step
        self.prev_subs: Optional[List[float]] = None
        # Previous fwd_return_mean, to detect directional move
        self.prev_mu: Optional[float] = None

    def update_and_get_weights(
        self, subs: List[float], mu: float
    ) -> List[float]:
        """Update hit-rate EMAs using past info only, return normalised weights."""
        if self.prev_subs is not None and self.prev_mu is not None:
            delta_mu = mu - self.prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                for i, ps in enumerate(self.prev_subs):
                    if abs(ps) > _EPS:
                        hit = 1.0 if (ps * move_sign) > 0.0 else 0.0
                    else:
                        hit = 0.5  # flat sub-signal: neutral update
                    self.hit_rates[i] = (
                        _HR_ALPHA * hit
                        + (1.0 - _HR_ALPHA) * self.hit_rates[i]
                    )

        # Store current for next step (strictly past info from next step's view)
        self.prev_subs = list(subs)
        self.prev_mu   = mu

        # Weights = max(0, hit_rate - 0.5); only reward better-than-coin-flip
        raw_weights = [max(0.0, hr - 0.5) for hr in self.hit_rates]
        total = sum(raw_weights)
        if total < _EPS:
            # All sub-signals at or below coin-flip: equal weight (will be small)
            return [1.0 / _N_SUBS] * _N_SUBS
        return [w / total for w in raw_weights]


class PerfWeightedMetaStrategy:
    """Hit-rate-weighted blend of drift, carry, and reversion sub-signals."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Sub-signals (same basis as adaptive_blend)
        drift = math.tanh(mu / (sig + _EPS))
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)
        reversion = -drift

        subs = [drift, carry, reversion]

        sym_state = self._get_state(state.symbol)
        weights = sym_state.update_and_get_weights(subs, mu)

        composite = sum(w * s for w, s in zip(weights, subs))

        confidence = max(
            0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score))
        )

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=composite,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh PerfWeightedMetaStrategy instance."""
    return PerfWeightedMetaStrategy()
