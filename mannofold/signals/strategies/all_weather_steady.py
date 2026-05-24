"""All-weather steady strategy: engineered for LOW monthly return variance.

CONCEPT: Blend trend + carry + (calm-only) reversion, each vol-normalised,
gated by density, with a modest exposure cap, heavy EMA smoothing, and
entry/exit hysteresis — maximising the fraction of positive months across
ALL regimes rather than peak Sharpe.

  trend     = tanh(mu / (sig + eps))               Sharpe-based direction
  carry     = tanh(clamp(mu / (sig^2+eps)))         risk-adjusted carry
  reversion = -trend * calm_gate                    counter-trend in calm only
  composite = W_T*trend + W_C*carry + W_R*reversion

  confidence   = regime_prob * (1 - anomaly_score)
  density_gate = logistic(density)                  gates sizing by typicality
  weight       = clamp(tanh(GAIN*composite), -0.6, 0.6)
                   * density_gate * confidence

Heavy EMA (alpha=0.12) keeps month-over-month changes small.
Hysteresis: entry at |w|>=0.12, exit when |w|<0.05.
Flat on ANOMALY_REGIME or anomaly_score > 0.60.
Dead-band |w| < 0.04 -> 0.
Per-symbol state; build() takes no arguments; no lookahead.
"""

from __future__ import annotations

import math
from typing import Dict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "all_weather_steady"
DESCRIPTION = (
    "All-weather book targeting the steadiest monthly returns: trend + carry + "
    "calm-only reversion, vol-normalised, density-gated, heavily EMA-smoothed "
    "with hysteresis and modest exposure cap for low monthly return variance."
)

# sub-signal blend weights
_W_TREND  = 0.40
_W_CARRY  = 0.35
_W_REV    = 0.25   # reversion only active in calm (high-density) states

# sizing
_GAIN       = 2.2
_MAX_WEIGHT = 0.60  # modest cap for monthly consistency

# carry
_CARRY_CLAMP = 5.0

# density gate (logistic): high density -> gate~1, low stress -> gate~0
_DENSITY_MID   = 1.0
_DENSITY_SCALE = 2.5
_DENSITY_CLAMP = 50.0

# heavy EMA smoothing — primary consistency mechanism
_EMA_ALPHA = 0.12   # low alpha = slow/smooth, small month-over-month swings

# hysteresis
_ENTRY_THRESH = 0.12
_EXIT_THRESH  = 0.05

# gates
_ANOMALY_THRESH = 0.60
_DEAD_BAND      = 0.04
_EPS            = 1e-9


def _density_gate(density: float) -> float:
    """Smooth gate in [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class AllWeatherSteadyStrategy:
    """Trend + carry + calm-only reversion, heavily smoothed for monthly consistency."""

    def __init__(self) -> None:
        self._ema: Dict[str, float] = {}
        self._stance: Dict[str, int] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # Trend: Sharpe-based direction
        trend = math.tanh(mu / (sig + _EPS))

        # Carry: risk-adjusted return ratio, clamped then squashed
        carry_raw = mu / (sig ** 2 + _EPS)
        carry = math.tanh(max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw)))

        # Reversion: counter-trend, active ONLY in calm (high-density) states
        dgate = _density_gate(state.density)
        reversion = -trend * dgate  # dgate~1 in calm -> active; ~0 in stress -> off

        composite = _W_TREND * trend + _W_CARRY * carry + _W_REV * reversion

        # Heavy EMA smoothing — the primary monthly-variance-reduction mechanism
        sym = state.symbol
        prev = self._ema.get(sym, composite)
        smoothed = _EMA_ALPHA * composite + (1.0 - _EMA_ALPHA) * prev
        self._ema[sym] = smoothed

        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=smoothed,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gate: anomalous regime or high anomaly -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Sizing: tanh(gain*composite) capped, then scaled by confidence
        raw = math.tanh(_GAIN * signals.momentum)
        raw = max(-_MAX_WEIGHT, min(_MAX_WEIGHT, raw)) * signals.confidence

        # Dead-band: suppress sub-threshold noise
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Hysteresis
        stance = self._stance.get(sym, 0)
        desired = 1 if raw > 0 else (-1 if raw < 0 else 0)

        if stance == 0:
            if abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            if desired != 0 and desired != stance and abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired
            elif abs(raw) < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        weight = abs(raw) * self._stance[sym]
        weight = max(-_MAX_WEIGHT, min(_MAX_WEIGHT, weight))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh AllWeatherSteadyStrategy instance."""
    return AllWeatherSteadyStrategy()
