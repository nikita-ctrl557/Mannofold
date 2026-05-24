"""Multi-armed bandit strategy: softmax bandit over three arm signals.

Arms:
  momentum  =  +tanh(sharpe)
  reversion =  -tanh(sharpe)
  carry     =  tanh(clamp(mu / (sig^2 + eps)))

Per-symbol EMA rewards track past directional accuracy. Rewards are updated
BEFORE selection using only past information (strictly no lookahead). Best arm
(argmax softmax) is selected; signal scaled by confidence=regime_prob*(1-anomaly).
Flat on ANOMALY_REGIME or anomaly > 0.6.
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

NAME = "bandit_selector"
DESCRIPTION = (
    "Softmax multi-armed bandit that selects among momentum, reversion, and carry "
    "arms using online EMA reward tracking with strictly causal updates."
)

_EPS            = 1e-9
_CARRY_CLAMP    = 5.0
_GAIN           = 2.5
_EMA_ALPHA      = 0.10
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04
_N_ARMS         = 3   # momentum, reversion, carry
_SOFTMAX_TEMP   = 1.0


def _softmax(vals: List[float], temp: float = _SOFTMAX_TEMP) -> List[float]:
    scaled = [v / temp for v in vals]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    s = sum(exps)
    return [e / s for e in exps]


class _SymbolState:
    """Per-symbol bandit state with EMA reward tracking."""

    def __init__(self) -> None:
        self.ema_rewards: List[float] = [0.0] * _N_ARMS
        self.prev_arms: Optional[List[float]] = None
        self.prev_mu: Optional[float] = None

    def update_rewards_and_select(
        self, arms: List[float], mu: float
    ) -> int:
        """Update EMA rewards using PAST info, then return best arm index."""
        # Update using previous step's arm signals vs realized move now
        if self.prev_arms is not None and self.prev_mu is not None:
            delta_mu = mu - self.prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                for i, pa in enumerate(self.prev_arms):
                    if abs(pa) > _EPS:
                        reward = 1.0 if (pa * move_sign) > 0.0 else -1.0
                    else:
                        reward = 0.0
                    self.ema_rewards[i] = (
                        _EMA_ALPHA * reward
                        + (1.0 - _EMA_ALPHA) * self.ema_rewards[i]
                    )

        # Store current arms for next step (past info from next step's view)
        self.prev_arms = list(arms)
        self.prev_mu = mu

        # Pick best arm via argmax of softmax probabilities (deterministic)
        probs = _softmax(self.ema_rewards)
        best = probs.index(max(probs))
        return best


class BanditSelectorStrategy:
    """Softmax multi-armed bandit strategy selecting among momentum/reversion/carry."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Sharpe-based signal for momentum/reversion arms
        sharpe = mu / (sig + _EPS)

        # Three arm signals
        momentum  = math.tanh(sharpe)
        reversion = -math.tanh(sharpe)
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)

        arms = [momentum, reversion, carry]

        sym_state = self._get_state(state.symbol)
        best_arm = sym_state.update_rewards_and_select(arms, mu)

        selected_signal = arms[best_arm]

        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=selected_signal,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh BanditSelectorStrategy instance."""
    return BanditSelectorStrategy()
