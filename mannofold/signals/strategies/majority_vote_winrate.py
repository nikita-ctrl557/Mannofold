"""Majority-vote win-rate strategy: four independent voters, unanimous (4/4) required.

Voters: (1) sign(drift), (2) sign(EMA-drift, per-symbol, no lookahead),
        (3) sign(carry = mu/(sig²+eps)), (4) density-confidence gate
        (votes with sign(drift) only when density AND regime_prob are both high).

Require UNANIMOUS 4/4 AND |sharpe| > _SHARPE_MIN; otherwise flat.
weight = consensus_dir * tanh(GAIN*tanh(sharpe)) * (n_agree/4) * confidence
confidence = regime_prob * (1 - anomaly_score)
Flat on ANOMALY_REGIME, anomaly > 0.6, or weak agreement. Dead-band |w|<0.04->0.
"""
from __future__ import annotations
import math
from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import ANOMALY_REGIME, ManifoldState, SignalSet, TargetPosition

NAME = "majority_vote_winrate"
DESCRIPTION = (
    "Four-voter UNANIMOUS (4/4) consensus; voters: drift-sign, EMA-drift-sign, "
    "carry-sign, density-confidence gate. Only acts when all four agree AND "
    "|Sharpe|>=threshold — targets high win rate with positive expectancy."
)

_EPS = 1e-9
_GAIN = 2.5
_ANOMALY_THRESH = 0.6
_DEAD_BAND = 0.04
_EMA_ALPHA = 0.05          # slow EMA for stability, fewer sign-flips
_MIN_AGREE = 4             # require ALL four voters (unanimous)
_SHARPE_MIN = 0.3          # quality gate: |sharpe| must exceed this
_DENSITY_MID = 0.5         # sigmoid mid-point (density range ~0.18..0.77)
_DENSITY_SCALE = 5.0
_DENSITY_CLAMP = 50.0
_REGIME_PROB_THRESH = 0.6  # min regime_prob for voter 4 to cast a vote


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _density_gate(density: float) -> float:
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class MajorityVoteWinRateStrategy:
    """Four-voter unanimous strategy targeting high hit rate and positive expectancy."""

    def __init__(self) -> None:
        self._ema: dict[str, float] = {}  # per-symbol EMA of fwd_return_mean

    def signals(self, state: ManifoldState) -> SignalSet:
        mu, sig, sym = state.fwd_return_mean, state.fwd_return_std, state.symbol

        # Per-symbol EMA drift (online, no lookahead)
        self._ema[sym] = _EMA_ALPHA * mu + (1.0 - _EMA_ALPHA) * self._ema.get(sym, mu)
        ema_drift = self._ema[sym]

        # Four voters
        v1 = _sign(mu)                              # instantaneous drift
        v2 = _sign(ema_drift)                       # smoothed drift
        v3 = _sign(mu / (sig * sig + _EPS))         # carry = mu / variance
        gate = _density_gate(state.density)
        v4 = _sign(mu) if (gate > 0.5 and state.regime_prob > _REGIME_PROB_THRESH) else 0

        n_pos = sum(1 for v in [v1, v2, v3, v4] if v > 0)
        n_neg = sum(1 for v in [v1, v2, v3, v4] if v < 0)

        if n_pos >= _MIN_AGREE:
            consensus_dir, n_agree = 1, n_pos
        elif n_neg >= _MIN_AGREE:
            consensus_dir, n_agree = -1, n_neg
        else:
            consensus_dir, n_agree = 0, 0

        sharpe = mu / (sig + _EPS)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        if consensus_dir != 0 and abs(sharpe) >= _SHARPE_MIN:
            composite = consensus_dir * abs(math.tanh(_GAIN * math.tanh(sharpe))) * (n_agree / 4.0)
        else:
            composite = 0.0

        return SignalSet(ts=state.ts, symbol=sym, momentum=composite,
                        expected_return=mu, anomaly=state.anomaly_score,
                        regime_id=state.regime_id, confidence=confidence)

    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)
        raw = signals.momentum * signals.confidence
        if abs(raw) < _DEAD_BAND:
            raw = 0.0
        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MajorityVoteWinRateStrategy instance."""
    return MajorityVoteWinRateStrategy()
