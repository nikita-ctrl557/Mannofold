"""Regime Specialist Router: learns per-regime whether to FOLLOW or FADE.

For each observed regime_id, maintains two EMA reward trackers:
  - FOLLOW policy: +tanh(sharpe) direction
  - FADE   policy: -tanh(sharpe) direction

At each step, the previous step's chosen policy direction is compared to the
realized move (sign of delta in fwd_return_mean). The matching policy's EMA
reward is updated. The policy with the higher EMA reward is then selected.

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

NAME = "regime_specialist_router"
DESCRIPTION = (
    "Per-regime online learner that selects FOLLOW or FADE policy based on "
    "EMA reward tracking; emits direction * tanh(gain*tanh(sharpe)) * confidence."
)

_EPS             = 1e-9
_GAIN            = 2.5
_EMA_ALPHA       = 0.10   # EMA decay for reward updates
_ANOMALY_THRESH  = 0.6
_DEAD_BAND       = 0.04

# Policy indices
_FOLLOW = 0
_FADE   = 1


class _RegimeState:
    """EMA reward state for one regime's two policies (FOLLOW / FADE)."""

    def __init__(self) -> None:
        # Start equal so selection is random initially
        self.ema_rewards: list[float] = [0.0, 0.0]
        # Stored from previous step: (policy_index, signed_direction, prev_mu)
        self.prev: Optional[Tuple[int, float, float]] = None

    def update_and_select(self, mu: float) -> Tuple[int, float]:
        """Update EMA rewards from previous step, then select best policy.

        Returns (policy_index, chosen_direction) where direction is +1 or -1.
        Update uses ONLY past information (the previous step's action vs current mu).
        """
        # --- update from previous step ---
        if self.prev is not None:
            prev_policy, prev_dir, prev_mu = self.prev
            delta_mu = mu - prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                # +1 reward if prev_dir matched move, -1 otherwise
                reward = 1.0 if (prev_dir * move_sign) > 0.0 else -1.0
                self.ema_rewards[prev_policy] = (
                    _EMA_ALPHA * reward
                    + (1.0 - _EMA_ALPHA) * self.ema_rewards[prev_policy]
                )

        # --- select best policy ---
        if self.ema_rewards[_FOLLOW] >= self.ema_rewards[_FADE]:
            policy = _FOLLOW
        else:
            policy = _FADE

        # FOLLOW: go with the drift direction; FADE: go against it
        sharpe = mu / (abs(mu) + _EPS)  # sign of mu (±1 near extremes)
        # direction is +1 (long) or -1 (short) based on policy + drift sign
        sharpe_full = mu  # raw drift for tanh computation below
        drift_sign = 1.0 if sharpe_full >= 0.0 else -1.0
        direction = drift_sign if policy == _FOLLOW else -drift_sign

        # Store for next step's update
        self.prev = (policy, direction, mu)

        return policy, direction


class RegimeSpecialistRouter:
    """Routes per regime to FOLLOW or FADE policy, learning online."""

    def __init__(self) -> None:
        # Per-symbol, per-regime state: (symbol, regime_id) -> _RegimeState
        self._states: Dict[Tuple[str, int], _RegimeState] = {}
        # Store previous mu per symbol for reward signal
        self._prev_mu: Dict[str, float] = {}

    def _get_state(self, symbol: str, regime_id: int) -> _RegimeState:
        key = (symbol, regime_id)
        if key not in self._states:
            self._states[key] = _RegimeState()
        return self._states[key]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        sharpe = mu / (sig + _EPS)

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        # Flat signal when anomalous
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

        # Final signal: direction * tanh(gain * tanh(sharpe))
        inner = math.tanh(_GAIN * math.tanh(sharpe))
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
    """Return a fresh RegimeSpecialistRouter instance."""
    return RegimeSpecialistRouter()
