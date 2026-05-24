"""Adaptive-blend strategy: ensemble of drift, carry, and reversion sub-signals
with online EMA weight adaptation based on realized directional hit-rate.

Sub-signals:
  drift    =  tanh(mu / (sig + eps))
  carry    =  tanh(clamp(mu / (sig^2 + eps)))
  reversion = -drift

Weights are maintained per-symbol as EMA rewards (sign-match of previous
sub-signal direction vs current observed move). Rewards are softmax-normalised
to weights each step. Only past information is used — no lookahead.
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

NAME = "adaptive_blend"
DESCRIPTION = "Ensemble of drift, carry, and reversion sub-signals with EMA-based online weight adaptation."

_EPS           = 1e-9
_CARRY_CLAMP   = 5.0
_GAIN          = 2.5
_EMA_ALPHA     = 0.10   # EMA decay for reward tracking (lower = longer memory)
_ANOMALY_THRESH = 0.6
_DEAD_BAND     = 0.04
_N_SUBS        = 3      # drift, carry, reversion


def _softmax(vals: List[float]) -> List[float]:
    mx = max(vals)
    exps = [math.exp(v - mx) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


class _SymbolState:
    """Per-symbol online learning state."""

    def __init__(self) -> None:
        # EMA reward per sub-signal (init equal so softmax starts uniform)
        self.ema_rewards: List[float] = [0.0] * _N_SUBS
        # Previous sub-signal directions (+1 / -1 / 0), to compare next step
        self.prev_subs: Optional[List[float]] = None
        # Previous fwd_return_mean, to detect directional move
        self.prev_mu: Optional[float] = None

    def update_and_get_weights(
        self, subs: List[float], mu: float
    ) -> List[float]:
        """Update EMA rewards with past-vs-current comparison, return weights."""
        if self.prev_subs is not None and self.prev_mu is not None:
            # Realised move: sign of change in fwd_return_mean
            delta_mu = mu - self.prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                for i, ps in enumerate(self.prev_subs):
                    # Reward = +1 if sign matches, -1 if opposite, 0 if flat
                    if abs(ps) > _EPS:
                        reward = 1.0 if (ps * move_sign) > 0.0 else -1.0
                    else:
                        reward = 0.0
                    self.ema_rewards[i] = (
                        _EMA_ALPHA * reward
                        + (1.0 - _EMA_ALPHA) * self.ema_rewards[i]
                    )

        # Store current for next step (strictly past info from next step's view)
        self.prev_subs = list(subs)
        self.prev_mu   = mu

        return _softmax(self.ema_rewards)


class AdaptiveBlendStrategy:
    """Adaptive-weighted blend of drift, carry, and reversion sub-signals."""

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

        # Sub-signals
        drift = math.tanh(mu / (sig + _EPS))
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)
        reversion = -drift

        subs = [drift, carry, reversion]

        sym_state = self._get_state(state.symbol)
        weights = sym_state.update_and_get_weights(subs, mu)

        composite = sum(w * s for w, s in zip(weights, subs))

        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

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
    """Return a fresh AdaptiveBlendStrategy instance."""
    return AdaptiveBlendStrategy()
