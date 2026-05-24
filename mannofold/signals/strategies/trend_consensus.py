"""Trend-consensus strategy: majority vote across three independent trend voters.

Three voters:
  (1) Instantaneous drift sign  = sign(fwd_return_mean)
  (2) Per-symbol EMA-of-drift sign (smoothed view, no lookahead)
  (3) Sharpe sign               = sign(fwd_return_mean / (fwd_return_std + eps))

Take a position only when >=2 of 3 agree. Size by consensus strength:
  weight = consensus_dir * (n_agree/3) * tanh(gain * tanh(sharpe)) * confidence
where confidence = regime_prob * (1 - anomaly_score).
Flat on ANOMALY_REGIME, anomaly > 0.6, or no majority. Dead-band |w| < 0.04 -> 0.
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

NAME = "trend_consensus"
DESCRIPTION = (
    "Majority vote across drift-sign, EMA-drift-sign, and Sharpe-sign voters; "
    "sized by consensus strength and regime confidence."
)

_EPS = 1e-9
_GAIN = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND = 0.04
_EMA_ALPHA = 0.1   # smoothing factor for per-symbol EMA of drift


def _sign(x: float) -> int:
    """Return +1, -1, or 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


class TrendConsensusStrategy:
    """Three-voter trend consensus with per-symbol EMA state."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean (no lookahead — updated each bar)
        self._ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std
        sym = state.symbol

        # Update per-symbol EMA of drift (online, no lookahead)
        if sym not in self._ema:
            self._ema[sym] = mu
        else:
            self._ema[sym] = _EMA_ALPHA * mu + (1.0 - _EMA_ALPHA) * self._ema[sym]
        ema_drift = self._ema[sym]

        # ---- Three voters ----
        v1 = _sign(mu)              # instantaneous drift sign
        v2 = _sign(ema_drift)       # smoothed drift sign
        sharpe = mu / (sig + _EPS)
        v3 = _sign(sharpe)          # Sharpe sign

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

        # Encode consensus information into SignalSet fields:
        #   momentum  <- consensus_dir * (n_agree/3) * tanh(gain * tanh(sharpe))
        #   confidence <- confidence (as-is)
        if consensus_dir != 0:
            strength = (n_agree / 3.0) * math.tanh(_GAIN * math.tanh(sharpe))
            composite = consensus_dir * abs(strength)
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
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # composite stored in signals.momentum; scale by confidence
        raw = signals.momentum * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh TrendConsensusStrategy instance."""
    return TrendConsensusStrategy()
