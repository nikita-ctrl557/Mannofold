"""Mutual Information Strategy.

Information-theoretic strategy that estimates the online mutual information
I(signal_sign; move_sign) between the SIGN of the drift signal and the SIGN
of the subsequently-observed move, using a 2x2 contingency table of counts
updated with EMA decay.

Strictly no-lookahead: the signal sign from step t-1 is compared against
the move observed at step t (sign of change in fwd_return_mean).

MI = Σ p(x,y)·log(p(x,y)/(p(x)·p(y)))

HIGH MI → signal is genuinely predictive → size up.
LOW  MI → no edge                        → go flat.

target_weight = sign(sharpe) · tanh(gain · |tanh(sharpe)|)
              · clamp(MI / MI_max, 0, 1)
              · confidence

confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > ~0.6.
Dead-band |w| < 0.04 → 0.
Per-symbol state, no lookahead.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "mutual_information"
DESCRIPTION = (
    "Estimates online mutual information I(signal_sign; move_sign) via an "
    "EMA-decayed 2x2 contingency table (strictly past-only, no lookahead). "
    "HIGH MI → signal is predictive → size up; LOW MI → flat. "
    "Weight = sign(sharpe)·tanh(gain·|tanh(sharpe)|)·clamp(MI/MI_max,0,1)·confidence."
)

_EPS             = 1e-9
_GAIN            = 2.5    # scales tanh(sharpe) → magnitude
_MI_MAX          = 0.15   # normalisation ceiling for MI gate (lower = easier to open)
_EMA_ALPHA       = 0.12   # EMA decay for contingency table counts (higher = faster adapt)
_ANOMALY_THRESH  = 0.6
_DEAD_BAND       = 0.04
_MIN_COUNTS      = 2.0    # minimum total pseudo-count before trusting MI


class _SymbolState:
    """Per-symbol EMA contingency table for MI estimation.

    Table cells: counts[signal_sign_idx][move_sign_idx]
      signal_sign: 0 = negative/zero, 1 = positive
      move_sign:   0 = negative/zero, 1 = positive

    We track EMA-weighted counts so old data is forgotten.
    """

    __slots__ = ("counts", "prev_signal_sign", "prev_mu", "total")

    def __init__(self) -> None:
        # 2x2 table, initialised with a tiny uniform prior
        self.counts: list[list[float]] = [[0.25, 0.25], [0.25, 0.25]]
        self.total: float = 1.0          # sum of all counts
        self.prev_signal_sign: Optional[int] = None
        self.prev_mu: Optional[float] = None

    def _decay(self) -> None:
        """Apply EMA decay to all counts."""
        decay = 1.0 - _EMA_ALPHA
        total = 0.0
        for i in range(2):
            for j in range(2):
                self.counts[i][j] *= decay
                total += self.counts[i][j]
        self.total = total

    def update(self, signal_sign: int, mu: float) -> float:
        """Update table with previous signal vs current move; return MI estimate.

        The update uses: PREVIOUS step's signal_sign vs CURRENT move direction.
        No lookahead: current signal_sign is stored for the NEXT step's update.
        """
        mi = 0.0

        if self.prev_signal_sign is not None and self.prev_mu is not None:
            delta = mu - self.prev_mu
            move_sign = 1 if delta > _EPS else 0

            # EMA decay, then increment
            self._decay()
            self.counts[self.prev_signal_sign][move_sign] += _EMA_ALPHA * self.total + _EPS
            self.total = sum(self.counts[i][j] for i in range(2) for j in range(2))

            # Compute MI only when we have enough data
            if self.total >= _MIN_COUNTS:
                mi = self._compute_mi()

        # Store current signal for next step
        self.prev_signal_sign = signal_sign
        self.prev_mu = mu
        return mi

    def _compute_mi(self) -> float:
        """Compute MI = Σ p(x,y)·log(p(x,y)/(p(x)·p(y))) from current counts."""
        t = max(self.total, _EPS)
        # Joint probs
        p = [[self.counts[i][j] / t for j in range(2)] for i in range(2)]
        # Marginals
        px = [p[i][0] + p[i][1] for i in range(2)]
        py = [p[0][j] + p[1][j] for j in range(2)]

        mi = 0.0
        for i in range(2):
            for j in range(2):
                pxy = p[i][j]
                denom = max(px[i] * py[j], _EPS)
                if pxy > _EPS:
                    mi += pxy * math.log(pxy / denom)
        return max(0.0, mi)


class MutualInformationStrategy:
    """Per-symbol online mutual-information gated strategy."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        std = max(state.fwd_return_std, _EPS)

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            return SignalSet(
                ts=state.ts, symbol=state.symbol,
                momentum=0.0, expected_return=mu,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id, confidence=0.0,
            )

        # Current signal direction (used to update table next step)
        sharpe = mu / (std + _EPS)
        signal_sign = 1 if sharpe > 0.0 else 0

        sym = self._get_state(state.symbol)
        mi = sym.update(signal_sign, mu)

        # MI gate: clamp(MI / MI_max, 0, 1)
        mi_gate = min(1.0, max(0.0, mi / max(_MI_MAX, _EPS)))

        # Direction: sign(sharpe) · tanh(gain · |tanh(sharpe)|)
        sign_s = 1.0 if sharpe >= 0.0 else -1.0
        magnitude = math.tanh(_GAIN * abs(math.tanh(sharpe)))
        momentum = sign_s * magnitude * mi_gate

        return SignalSet(
            ts=state.ts, symbol=state.symbol,
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
    """Return a fresh MutualInformationStrategy instance."""
    return MutualInformationStrategy()
