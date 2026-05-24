"""Monthly Momentum Persistence: ride states that have been consistently rewarding.

Maintains a per-symbol EMA of realized returns to the chosen direction (past-only).
Scales exposure UP when this reward EMA is positive and stable, DOWN when negative.
The engine leans in during good runs and pulls back in bad ones, smoothing month-to-month.

weight = sign(sharpe)*tanh(gain*|tanh(sharpe)|)*clamp(0.2+2*reward_ema,0,1)*confidence
confidence = regime_prob*(1-anomaly_score)
Flat on ANOMALY_REGIME or anomaly>0.6; dead-band |w|<0.04->0; per-symbol state; no lookahead.
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

NAME = "monthly_momentum_persist"
DESCRIPTION = (
    "Month-over-month consistency engine: EMA reward of realized returns scales exposure "
    "up during good runs and down during bad ones; "
    "weight = sign(sharpe)*tanh(gain*|tanh(sharpe)|)*clamp(0.2+2*ema,0,1)*confidence."
)

# Tunable knobs
_GAIN            = 2.5    # amplification inside tanh for sharpe-based sizing
_EMA_ALPHA       = 0.05   # slow EMA (~20-bar half-life, ~monthly at daily bars)
_ANOMALY_THRESH  = 0.6    # anomaly_score above this -> flat
_DEAD_BAND       = 0.04   # collapse |weight| below this to 0
_EPS             = 1e-9


class MonthlyMomentumPersist:
    """Per-symbol EMA reward tracker that scales exposure by recent consistency."""

    def __init__(self) -> None:
        # Per-symbol: EMA of signed realized reward (+1 if direction matched move, -1 if not)
        self._reward_ema: dict[str, float] = {}
        # Per-symbol: chosen direction from previous step (for reward calculation)
        self._prev_direction: dict[str, float] = {}
        # Per-symbol: previous fwd_return_mean (to measure delta / realized move)
        self._prev_mu: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Confidence: regime certainty × non-anomalousness
        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        # Flat signal when anomalous
        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            # Still update state for non-anomalous bookkeeping, but emit flat
            self._prev_mu[sym] = mu
            return SignalSet(
                ts=state.ts,
                symbol=sym,
                momentum=0.0,
                expected_return=mu,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=0.0,
            )

        # Update reward EMA from the previous step's direction vs realized move
        prev_dir = self._prev_direction.get(sym, 0.0)
        prev_mu  = self._prev_mu.get(sym, mu)
        ema      = self._reward_ema.get(sym, 0.0)

        delta_mu = mu - prev_mu
        if abs(delta_mu) > _EPS and prev_dir != 0.0:
            move_sign = 1.0 if delta_mu > 0.0 else -1.0
            reward = 1.0 if (prev_dir * move_sign) > 0.0 else -1.0
            ema = _EMA_ALPHA * reward + (1.0 - _EMA_ALPHA) * ema

        self._reward_ema[sym] = ema
        self._prev_mu[sym]    = mu

        # Sharpe-based direction and magnitude
        sharpe = mu / (sig + _EPS)
        sharpe_sign = 1.0 if sharpe >= 0.0 else -1.0

        # Saturating magnitude: tanh(gain * |tanh(sharpe)|)
        magnitude = math.tanh(_GAIN * abs(math.tanh(sharpe)))

        # Persistence scalar: clamp(0.2 + 2*reward_ema, 0, 1)
        persistence = max(0.0, min(1.0, 0.2 + 2.0 * ema))

        # Full momentum signal (before confidence)
        momentum = sharpe_sign * magnitude * persistence

        # Record chosen direction for next step's reward calculation
        self._prev_direction[sym] = sharpe_sign

        return SignalSet(
            ts=state.ts,
            symbol=sym,
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

        # Dead-band: suppress small, noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MonthlyMomentumPersist instance."""
    return MonthlyMomentumPersist()
