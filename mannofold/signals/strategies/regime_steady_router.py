"""Regime Steady Router: favours the STEADIEST policy per regime.

For each observed regime_id, tracks online (past-only) both the mean reward
AND the variance of reward for two policies (FOLLOW / FADE).

At each step, the Sharpe-like ratio (mean / std) is computed for each policy
and the one with the better risk-adjusted performance is chosen.

Final weight = chosen_dir * tanh(gain * tanh(sharpe)) * confidence
  where confidence = regime_prob * (1 - anomaly_score).

Flat on ANOMALY_REGIME or anomaly_score > 0.6. Dead-band |w| < 0.04 -> 0.
Deterministic, no lookahead.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "regime_steady_router"
DESCRIPTION = (
    "Per-regime online learner that selects FOLLOW or FADE based on the best "
    "mean/variance (Sharpe-like) ratio, favouring month-over-month consistency "
    "over high-variance policies."
)

_EPS            = 1e-9
_GAIN           = 2.5
_VAR_ALPHA      = 0.10   # EMA decay for mean/variance tracking
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04

_FOLLOW = 0
_FADE   = 1


class _PolicyStats:
    """Online Welford-style EMA mean and variance tracker for one policy."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float  = 1.0   # initialise > 0 so Sharpe starts at 0
        self.n: int      = 0

    def update(self, reward: float) -> None:
        """Exponential moving mean + variance update (no lookahead)."""
        self.n += 1
        delta = reward - self.mean
        self.mean = _VAR_ALPHA * reward + (1.0 - _VAR_ALPHA) * self.mean
        delta2 = reward - self.mean
        self.var = (1.0 - _VAR_ALPHA) * (self.var + _VAR_ALPHA * delta * delta2)
        self.var = max(self.var, _EPS)

    def steady_sharpe(self) -> float:
        """Return mean / std — higher means steadier positive reward."""
        return self.mean / math.sqrt(self.var)


class _RegimeState:
    """Steady-policy state for one regime: tracks two policies' Sharpe."""

    def __init__(self) -> None:
        self.stats: list[_PolicyStats] = [_PolicyStats(), _PolicyStats()]
        # Previous step: (policy_index, direction, prev_mu)
        self.prev: Optional[Tuple[int, float, float]] = None

    def update_and_select(self, mu: float) -> Tuple[int, float]:
        """Update stats from previous step (past-only), then select best policy.

        Returns (policy_index, chosen_direction).
        """
        # --- update from previous step's outcome ---
        if self.prev is not None:
            prev_policy, prev_dir, prev_mu = self.prev
            delta_mu = mu - prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                reward = 1.0 if (prev_dir * move_sign) > 0.0 else -1.0
                self.stats[prev_policy].update(reward)

        # --- select policy with better Sharpe-like consistency ---
        sh_follow = self.stats[_FOLLOW].steady_sharpe()
        sh_fade   = self.stats[_FADE].steady_sharpe()

        if sh_follow >= sh_fade:
            policy = _FOLLOW
        else:
            policy = _FADE

        drift_sign = 1.0 if mu >= 0.0 else -1.0
        direction  = drift_sign if policy == _FOLLOW else -drift_sign

        self.prev = (policy, direction, mu)
        return policy, direction


class RegimeSteadyRouter:
    """Routes per regime to the policy with the best consistency (Sharpe-like)."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, int], _RegimeState] = {}

    def _get_state(self, symbol: str, regime_id: int) -> _RegimeState:
        key = (symbol, regime_id)
        if key not in self._states:
            self._states[key] = _RegimeState()
        return self._states[key]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        sharpe     = mu / (sig + _EPS)
        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            return SignalSet(
                ts=state.ts,
                symbol=state.symbol,
                momentum=0.0,
                expected_return=mu,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=0.0,
            )

        regime_state = self._get_state(state.symbol, state.regime_id)
        _policy, direction = regime_state.update_and_select(mu)

        inner    = math.tanh(_GAIN * math.tanh(sharpe))
        momentum = direction * inner

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
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
    """Return a fresh RegimeSteadyRouter instance."""
    return RegimeSteadyRouter()
