"""Majority-vote win-rate strategy.

Four independent voters:
  (1) sign(drift)             — instantaneous direction
  (2) sign(EMA-drift)         — smoothed per-symbol direction (no lookahead)
  (3) sign(carry)             — mu / (sig^2 + eps), risk-adjusted drift
  (4) density-confidence vote — votes with mu when density AND regime_prob are
                                both high enough; abstains (0) otherwise

Require UNANIMOUS 4/4 agreement AND |sharpe| > _SHARPE_MIN to take a position;
otherwise flat.  Strong unanimity + Sharpe threshold filters for high-quality
setups, targeting a high hit rate.

weight = consensus_dir * tanh(GAIN * tanh(sharpe)) * confidence
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME, anomaly > 0.6, or fewer than 4 voters agreeing,
or |sharpe| below threshold. Dead-band |w| < 0.04 -> 0.
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
    "Four-voter UNANIMOUS consensus (4/4 required); voters: drift-sign, "
    "EMA-drift-sign, carry-sign, density-confidence gate.  Only acts when all "
    "four voters agree AND |Sharpe| exceeds a threshold — targets high win rate "
    "with positive expectancy by restricting to the highest-conviction setups."
)

_EPS = 1e-9
_GAIN = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND = 0.04
_EMA_ALPHA = 0.05         # slow EMA — fewer sign flips, more stability
_MIN_AGREE = 4            # require ALL four voters to agree (unanimous)
_SHARPE_MIN = 0.3         # minimum |sharpe| to trade (quality filter)
# Density gate parameters (density range ~0.18..0.77 in practice)
_DENSITY_MID = 0.5        # sigmoid mid-point in the actual density range
_DENSITY_SCALE = 5.0      # steepness
_DENSITY_CLAMP = 50.0
_REGIME_PROB_THRESH = 0.6  # regime_prob threshold for voter 4 to cast a vote


def _sign(x: float) -> int:
    """Return +1, -1, or 0."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _density_gate(density: float) -> float:
    """Smooth sigmoid gate [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class MajorityVoteWinRateStrategy:
    """Four-voter unanimous strategy targeting high hit rate."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean (updated online, no lookahead)
        self._ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
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

        # ---- Four voters ----
        v1 = _sign(mu)                     # instantaneous drift sign
        v2 = _sign(ema_drift)              # smoothed EMA-drift sign
        carry = mu / (sig * sig + _EPS)
        v3 = _sign(carry)                  # carry = mu / variance

        # Voter 4: density-confidence gate
        # Votes in direction of instantaneous drift only when both density
        # and regime_prob are high enough to trust the environment.
        gate = _density_gate(state.density)
        if gate > 0.5 and state.regime_prob > _REGIME_PROB_THRESH:
            v4 = _sign(mu)   # trusted environment: align with drift
        else:
            v4 = 0           # low-confidence environment: abstain

        votes = [v1, v2, v3, v4]
        n_pos = sum(1 for v in votes if v > 0)
        n_neg = sum(1 for v in votes if v < 0)

        # Determine consensus — require unanimous (4/4)
        if n_pos >= _MIN_AGREE:
            consensus_dir = 1
            n_agree = n_pos
        elif n_neg >= _MIN_AGREE:
            consensus_dir = -1
            n_agree = n_neg
        else:
            consensus_dir = 0
            n_agree = 0

        # Sharpe for sizing
        sharpe = mu / (sig + _EPS)

        # Confidence = regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        # Composite: direction * tanh-squashed sharpe * agreement fraction
        # Only non-zero when unanimous AND sharpe passes quality threshold
        if consensus_dir != 0 and abs(sharpe) >= _SHARPE_MIN:
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
    """Return a fresh MajorityVoteWinRateStrategy instance."""
    return MajorityVoteWinRateStrategy()
