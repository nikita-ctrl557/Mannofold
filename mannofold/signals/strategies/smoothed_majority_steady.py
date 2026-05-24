"""Smoothed-majority-steady strategy: month-over-month consistency via heavy EMA.

Three voters determine direction (drift sign, EMA-drift sign, carry/Sharpe sign).
A position is taken only when >=2 of 3 agree.  The raw weight is then passed
through a VERY heavy per-symbol EMA (alpha~0.1) so exposure changes slowly,
producing low turnover and steady monthly returns.

  weight_raw = consensus_dir * (n_agree/3) * tanh(gain * tanh(sharpe)) * confidence
  smoothed   = alpha * weight_raw + (1 - alpha) * prev_smoothed   [alpha~0.1]
  confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME, anomaly > 0.6, or no majority.  Dead-band |w| < 0.04 -> 0.
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

NAME = "smoothed_majority_steady"
DESCRIPTION = (
    "Majority vote across drift-sign, EMA-drift-sign, and Sharpe/carry-sign voters "
    "with very heavy per-symbol EMA smoothing (alpha~0.1) for low-turnover, "
    "month-over-month consistent exposure."
)

_EPS = 1e-9
_GAIN = 2.5           # tanh gain — moderate conviction
_ANOMALY_THRESH = 0.6  # gate: go flat above this anomaly level
_DEAD_BAND = 0.04     # zero-out tiny positions to avoid noise trades
_EMA_ALPHA = 0.10     # very heavy smoothing: slow-moving targets = low turnover
_DRIFT_EMA_ALPHA = 0.10  # smoothing for the per-symbol drift EMA voter


def _sign(x: float) -> int:
    """Return +1, -1, or 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


class SmoothedMajoritySteadyStrategy:
    """Three-voter consensus with very heavy EMA smoothing for steady monthly returns."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean (voter 2 — no lookahead)
        self._drift_ema: dict[str, float] = {}
        # Per-symbol EMA of target weight (the heavy smoothing layer)
        self._weight_ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu = state.fwd_return_mean
        sig = state.fwd_return_std
        sym = state.symbol

        # Update per-symbol EMA of drift (online, no lookahead)
        if sym not in self._drift_ema:
            self._drift_ema[sym] = mu
        else:
            self._drift_ema[sym] = (
                _DRIFT_EMA_ALPHA * mu + (1.0 - _DRIFT_EMA_ALPHA) * self._drift_ema[sym]
            )
        ema_drift = self._drift_ema[sym]

        # ---- Three voters ----
        v1 = _sign(mu)                          # voter 1: instantaneous drift sign
        v2 = _sign(ema_drift)                   # voter 2: smoothed drift sign
        sharpe = mu / (sig + _EPS)
        v3 = _sign(sharpe)                      # voter 3: Sharpe / carry sign

        votes = [v1, v2, v3]
        n_pos = sum(1 for v in votes if v > 0)
        n_neg = sum(1 for v in votes if v < 0)

        if n_pos >= 2:
            consensus_dir = 1
            n_agree = n_pos
        elif n_neg >= 2:
            consensus_dir = -1
            n_agree = n_neg
        else:
            consensus_dir = 0
            n_agree = 0

        # Confidence = regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        # Composite signal stored in momentum field
        if consensus_dir != 0:
            strength = (n_agree / 3.0) * math.tanh(_GAIN * math.tanh(sharpe))
            composite = float(consensus_dir) * abs(strength)
        else:
            composite = 0.0

        return SignalSet(
            ts=state.ts,
            symbol=sym,
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
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat (decay EMA)
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            prev = self._weight_ema.get(sym, 0.0)
            self._weight_ema[sym] = (1.0 - _EMA_ALPHA) * prev
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: consensus * confidence
        raw = signals.momentum * signals.confidence

        # Very heavy EMA smoothing — this is what makes returns steady/low-turnover
        prev = self._weight_ema.get(sym, 0.0)
        smoothed = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * prev
        self._weight_ema[sym] = smoothed

        # Dead-band: zero out tiny positions
        weight = 0.0 if abs(smoothed) < _DEAD_BAND else smoothed

        # Hard clip to [-1, 1]
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh SmoothedMajoritySteadyStrategy instance."""
    return SmoothedMajoritySteadyStrategy()
