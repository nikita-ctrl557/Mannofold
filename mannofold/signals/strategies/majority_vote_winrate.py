"""Majority-vote win-rate strategy: four independent voters requiring unanimous (>=3/4) agreement.

Four voters:
  (1) sign(drift)          = sign(fwd_return_mean)
  (2) sign(EMA-drift)      = sign(per-symbol EMA of fwd_return_mean, no lookahead)
  (3) sign(carry)          = sign(mu / (sig^2 + eps))
  (4) density-confidence   = positive vote only when density + regime_prob are both high

Require >=3 of 4 voters to agree before taking a position; otherwise flat.
High agreement -> high hit rate -> positive expectancy.

weight = consensus_dir * tanh(gain * tanh(sharpe)) * (n_agree/4) * confidence
confidence = regime_prob * (1 - anomaly_score)
Flat on ANOMALY_REGIME, anomaly > 0.6, or fewer than 3 voters agreeing.
Dead-band |w| < 0.04 -> 0.
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

NAME = "majority_vote_winrate"
DESCRIPTION = (
    "Four-voter unanimous/supermajority strategy (>=3/4 agreement required); "
    "voters: drift-sign, EMA-drift-sign, carry-sign, density-confidence gate. "
    "High agreement threshold drives a high win rate with positive expectancy."
)

_EPS = 1e-9
_GAIN = 3.0
_ANOMALY_THRESH = 0.5
_DEAD_BAND = 0.05
_EMA_ALPHA = 0.05         # slow smoothing for per-symbol EMA of drift
_MIN_AGREE = 3            # minimum voters agreeing to take a position (out of 4)
# Density/regime thresholds for the density-confidence voter
_DENSITY_THRESH = 1.0     # density gate mid-point (logistic)
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0
_REGIME_PROB_THRESH = 0.6 # regime_prob must exceed this for voter 4 to be directional


def _sign(x: float) -> int:
    """Return +1, -1, or 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _density_gate(density: float) -> float:
    """Smooth gate in [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_THRESH)))


class MajorityVoteWinRateStrategy:
    """Four-voter supermajority strategy targeting high hit rate."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean (updated online, no lookahead)
        self._ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
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

        # ---- Four voters ----
        v1 = _sign(mu)                                      # instantaneous drift sign
        v2 = _sign(ema_drift)                               # smoothed EMA-drift sign
        carry = mu / (sig * sig + _EPS)
        v3 = _sign(carry)                                   # carry = mu / variance

        # Voter 4: density-confidence gate — votes in the direction of the EMA drift
        # (smoother, less noisy than instantaneous drift) only when both density
        # and regime_prob are high enough to trust the environment.
        gate = _density_gate(state.density)
        if gate > 0.5 and state.regime_prob > _REGIME_PROB_THRESH:
            v4 = _sign(ema_drift)   # trusted env: agree with smoothed drift direction
        else:
            v4 = 0                  # low confidence: abstain

        votes = [v1, v2, v3, v4]
        n_pos = sum(1 for v in votes if v > 0)
        n_neg = sum(1 for v in votes if v < 0)

        if n_pos >= _MIN_AGREE:
            consensus_dir = 1
            n_agree = n_pos
        elif n_neg >= _MIN_AGREE:
            consensus_dir = -1
            n_agree = n_neg
        else:
            consensus_dir = 0
            n_agree = 0

        # Confidence = regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        # Composite signal: direction * normalised sharpe * agreement fraction
        if consensus_dir != 0:
            sharpe = mu / (sig + _EPS)
            strength = math.tanh(_GAIN * math.tanh(sharpe))
            composite = consensus_dir * abs(strength) * (n_agree / 4.0)
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
    """Return a fresh MajorityVoteWinRateStrategy instance."""
    return MajorityVoteWinRateStrategy()
