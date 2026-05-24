"""Poisson jump-diffusion strategy: Merton jump-diffusion decomposition.

Separates market dynamics into two regimes:
  - DIFFUSION: normal drift — trade in the direction of the expected return.
  - JUMP: rare large move detected by a spike in anomaly_score or a large
    standardized fwd_return_mean. Jumps typically overshoot and partially
    revert, so we take a contrarian/fade position scaled by jump magnitude.

jump_flag = (z_score > Z_JUMP) OR (anomaly rising faster than FAST_RISE_THRESH)
target_weight = (jump? -fade_dir * tanh(GAIN * |z|) : +tanh(GAIN * tanh(sharpe)))
                * confidence
confidence = regime_prob * (1 - anomaly_score)
Flat on ANOMALY_REGIME; dead-band |w| < 0.04 -> 0.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "poisson_jump"
DESCRIPTION = (
    "Merton jump-diffusion decomposition: in the diffusion regime, trades the "
    "expected drift; on detected Poisson jumps (anomaly spike or large z-score "
    "move), fades the overshoot with a contrarian position scaled by jump magnitude."
)

# Tunable knobs
_GAIN = 2.5           # tanh amplifier: tanh(gain * |z|) or tanh(gain * tanh(sharpe))
_Z_JUMP = 2.0         # z-score threshold above which a jump is flagged
_FAST_RISE = 0.15     # anomaly delta threshold flagging a rapid jump onset
_WINDOW = 50          # rolling window for per-symbol mean/std of fwd_return_mean
_DEAD_BAND = 0.04     # |weight| below this is zeroed out


class PoissonJumpStrategy:
    """Per-symbol Merton jump-diffusion filter.

    Maintains a rolling window of fwd_return_mean to compute the per-symbol
    mean and std, enabling standardization into z-scores without lookahead.
    Also tracks the previous anomaly_score to measure the rate of rise.
    """

    def __init__(self) -> None:
        self._fwd_window: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_WINDOW)
        )
        self._prev_anomaly: dict[str, float] = {}
        # Per-symbol cache of signals from signals() for use in target()
        self._cache: dict[str, dict] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # --- Rolling window statistics for z-score (no lookahead) ---
        win = self._fwd_window[sym]
        if len(win) >= 2:
            n = len(win)
            mu = sum(win) / n
            variance = sum((v - mu) ** 2 for v in win) / n
            sigma = math.sqrt(variance) if variance > 0 else 1e-9
        else:
            mu = 0.0
            sigma = 1e-9
        # Update window AFTER computing stats (strict no-lookahead)
        win.append(state.fwd_return_mean)

        z = (state.fwd_return_mean - mu) / sigma

        # --- Anomaly rate of change ---
        prev_anom = self._prev_anomaly.get(sym, state.anomaly_score)
        anom_delta = state.anomaly_score - prev_anom
        self._prev_anomaly[sym] = state.anomaly_score

        # --- Jump detection (Poisson event flag) ---
        jump_flag = (abs(z) > _Z_JUMP) or (anom_delta > _FAST_RISE)

        # Direction of the jump move (for fade: we go opposite)
        fade_dir = math.copysign(1.0, z) if abs(z) > 1e-9 else 1.0

        # Confidence: shrink near anomalous or uncertain states
        confidence = max(0.0, min(1.0, state.regime_prob)) * max(
            0.0, 1.0 - state.anomaly_score
        )

        # --- Signal strength ---
        if jump_flag:
            # Contrarian: fade the jump overshoot
            raw_weight = -fade_dir * math.tanh(_GAIN * abs(z))
        else:
            # Diffusion: follow the drift
            sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
            raw_weight = math.tanh(_GAIN * math.tanh(sharpe))

        # Cache for target()
        self._cache[sym] = {
            "raw_weight": raw_weight,
            "confidence": confidence,
            "jump_flag": jump_flag,
        }

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=raw_weight,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard flat on anomalous regime sentinel
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        cache = self._cache.get(sym, {})
        raw_weight = cache.get("raw_weight", 0.0)
        confidence = cache.get("confidence", signals.confidence)

        weight = raw_weight * confidence

        # Dead-band suppression
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh PoissonJumpStrategy instance."""
    return PoissonJumpStrategy()
