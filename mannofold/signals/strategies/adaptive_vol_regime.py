"""Adaptive Vol-Regime Strategy: switches between trend and carry sub-signals
based on which has paid off recently AND the current volatility regime.

Sub-signals:
  trend  = tanh(mu / (sig + eps))          -- follow the drift direction
  carry  = tanh(clamp(mu / (sig^2 + eps))) -- Sharpe-normalised carry signal

Per-symbol EMA reward trackers are updated ONLY from past info: did the
previous step's sub-signal direction match the now-observed move (delta_mu)?

Vol regime prior: per-symbol EMA of fwd_return_std.
  high-vol => prior favours trend (score boost)
  low-vol  => prior favours carry (score boost)

Selection: choose sub-strategy with higher (ema_reward + vol_prior_boost).
weight = chosen_signal * confidence
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6. Dead-band |w| < 0.04 -> 0.
Deterministic, no lookahead. build() takes no arguments.
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

NAME = "adaptive_vol_regime"
DESCRIPTION = (
    "Adaptive strategy that switches between trend and carry sub-signals based on "
    "online EMA reward tracking and current volatility regime; high-vol favours trend, "
    "low-vol favours carry as a prior."
)

_EPS            = 1e-9
_CARRY_CLAMP    = 5.0
_EMA_ALPHA      = 0.10    # reward EMA decay
_VOL_EMA_ALPHA  = 0.05    # vol EMA decay (longer memory)
_VOL_PRIOR      = 0.15    # magnitude of vol-regime prior boost
_GAIN           = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04

# Sub-strategy indices
_TREND = 0
_CARRY = 1


class _SymbolState:
    """Per-symbol online state for the adaptive vol-regime strategy."""

    def __init__(self) -> None:
        self.ema_rewards: list[float] = [0.0, 0.0]  # [trend, carry]
        # previous step: (sub_signals, mu)
        self.prev: Optional[Tuple[list[float], float]] = None
        # EMA of fwd_return_std for vol regime detection
        self.vol_ema: float = 0.0
        self.vol_initialised: bool = False

    def step(self, subs: list[float], mu: float, sig: float) -> int:
        """Update EMA rewards from past info; update vol EMA; return chosen index."""
        # --- update reward EMAs using only past information ---
        if self.prev is not None:
            prev_subs, prev_mu = self.prev
            delta_mu = mu - prev_mu
            if abs(delta_mu) > _EPS:
                move_sign = 1.0 if delta_mu > 0.0 else -1.0
                for i, ps in enumerate(prev_subs):
                    if abs(ps) > _EPS:
                        reward = 1.0 if (ps * move_sign) > 0.0 else -1.0
                    else:
                        reward = 0.0
                    self.ema_rewards[i] = (
                        _EMA_ALPHA * reward
                        + (1.0 - _EMA_ALPHA) * self.ema_rewards[i]
                    )

        # --- update vol EMA (uses current sig, which is from frozen model — no lookahead) ---
        if not self.vol_initialised:
            self.vol_ema = sig
            self.vol_initialised = True
        else:
            self.vol_ema = _VOL_EMA_ALPHA * sig + (1.0 - _VOL_EMA_ALPHA) * self.vol_ema

        # --- vol-regime prior: high vol -> boost trend; low vol -> boost carry ---
        # compare current sig to running EMA: above => high-vol, below => low-vol
        if sig >= self.vol_ema:
            prior_boost = [_VOL_PRIOR, -_VOL_PRIOR]   # favour trend
        else:
            prior_boost = [-_VOL_PRIOR, _VOL_PRIOR]   # favour carry

        scores = [self.ema_rewards[i] + prior_boost[i] for i in range(2)]
        chosen = _TREND if scores[_TREND] >= scores[_CARRY] else _CARRY

        # store for next step
        self.prev = (list(subs), mu)

        return chosen


class AdaptiveVolRegimeStrategy:
    """Adaptive trend/carry switcher modulated by vol regime and EMA rewards."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Compute both sub-signals
        trend = math.tanh(mu / (sig + _EPS))
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)

        subs = [trend, carry]

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        # Flat signal when anomalous (still update state to avoid lookahead issues)
        sym_state = self._get_state(state.symbol)
        chosen_idx = sym_state.step(subs, mu, sig)

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

        chosen_signal = subs[chosen_idx]
        momentum = math.tanh(_GAIN * chosen_signal)

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
    """Return a fresh AdaptiveVolRegimeStrategy instance. No arguments required."""
    return AdaptiveVolRegimeStrategy()
