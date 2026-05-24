"""Consensus-Carry strategy: trend-consensus direction gating with Kelly carry sizing.

Three direction voters:
  (1) Instantaneous drift sign  = sign(fwd_return_mean)
  (2) Per-symbol EMA-drift sign (smoothed, no lookahead)
  (3) Sharpe sign               = sign(fwd_return_mean / (fwd_return_std + eps))

Require >=2/3 agreement for DIRECTION. Size MAGNITUDE by Kelly inverse-variance:
  |kelly| = |fwd_return_mean| / (fwd_return_std^2 + eps), clamped, then tanh-squashed.

target_weight = consensus_dir * tanh(gain * |kelly|) * (n_agree/3) * confidence
where confidence = regime_prob * (1 - anomaly_score).

Flat on ANOMALY_REGIME, anomaly > 0.6, or no majority. Dead-band |w| < 0.04 -> 0.
Per-symbol EMA on drift; no lookahead.
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

NAME = "consensus_carry"
DESCRIPTION = (
    "Combines trend-consensus direction voting (3 voters, >=2/3 majority) with "
    "Kelly inverse-variance carry sizing; confidence-weighted and regime-gated."
)

_EPS = 1e-9
_GAIN = 2.0
_KELLY_CAP = 3.0
_ANOMALY_THRESH = 0.6
_DEAD_BAND = 0.04
_EMA_ALPHA = 0.1


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


class ConsensusCarryStrategy:
    """Direction by majority vote; magnitude by Kelly carry; combined sizing."""

    def __init__(self) -> None:
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        mu = state.fwd_return_mean
        sig = state.fwd_return_std
        sym = state.symbol

        # Update per-symbol EMA of drift (online, no lookahead)
        if sym not in self._ema:
            self._ema[sym] = mu
        else:
            self._ema[sym] = _EMA_ALPHA * mu + (1.0 - _EMA_ALPHA) * self._ema[sym]
        ema_drift = self._ema[sym]

        # Three direction voters
        v1 = _sign(mu)                         # instant drift sign
        v2 = _sign(ema_drift)                  # EMA-drift sign
        v3 = _sign(mu / (sig + _EPS))          # Sharpe sign

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

        # Kelly magnitude: |mu| / sigma^2, clamped, tanh-squashed
        variance = sig ** 2 + _EPS
        kelly_raw = abs(mu) / variance
        kelly_clamped = min(kelly_raw, _KELLY_CAP)
        kelly_squashed = math.tanh(_GAIN * kelly_clamped)

        # Confidence
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        # Combined signal: direction * kelly_magnitude * vote_fraction * confidence
        if consensus_dir != 0:
            composite = consensus_dir * kelly_squashed * (n_agree / 3.0) * confidence
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

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = signals.momentum  # already includes confidence factor

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh ConsensusCarryStrategy instance."""
    return ConsensusCarryStrategy()
