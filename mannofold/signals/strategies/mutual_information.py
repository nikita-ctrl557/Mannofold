"""Mutual Information Strategy.

Information-theoretic strategy estimating online the mutual information
between the SIGN of the drift signal and the SIGN of the subsequently-
observed move, using a 2x2 contingency table updated with EMA decay.

Strictly no-lookahead: at each step we compare the PREVIOUS step's signal
sign vs the now-observed move sign, then update the contingency table.

MI = Σ p(x,y) · log( p(x,y) / (p(x)·p(y)) )

HIGH MI → the signal is genuinely predictive here → size up.
LOW MI  → no edge → go flat.

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
    "Information-theoretic strategy using online-estimated mutual information "
    "between the sign of the drift signal and the sign of the subsequently-observed "
    "move (2x2 contingency table with EMA decay). High MI → signal is predictive → "
    "size up; low MI → no edge → flat. "
    "Weight = sign(sharpe)·tanh(gain·|tanh(sharpe)|)·clamp(MI/MI_max,0,1)·confidence."
)

_EPS            = 1e-9
_GAIN           = 2.5    # scales tanh(sharpe) → momentum magnitude
_MI_MAX         = 0.15   # normalisation ceiling (max observed MI ≈ log(2) ≈ 0.693)
_EMA_DECAY      = 0.97   # per-step multiplicative decay of contingency counts
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04
_MIN_OBS        = 4      # steps before MI gate is trusted


class _SymbolState:
    """Per-symbol online state for MI estimation.

    Maintains a 2x2 contingency table (EMA-decayed pseudo-counts) of:
      rows: previous signal sign  (0 = non-positive, 1 = positive)
      cols: observed move sign    (0 = non-positive, 1 = positive)

    Storing prev_signal_sign + prev_mu enables strict no-lookahead updates:
    at step t we observe move = sign(mu_t - mu_{t-1}) and credit it against
    the signal sign that was known at step t-1.
    """

    __slots__ = ("counts", "prev_signal_sign", "prev_mu", "n_obs")

    def __init__(self) -> None:
        # Uniform prior: tiny equal counts to avoid log(0)
        self.counts: list[list[float]] = [[0.25, 0.25], [0.25, 0.25]]
        self.prev_signal_sign: Optional[int] = None
        self.prev_mu: Optional[float] = None
        self.n_obs: int = 0

    # ------------------------------------------------------------------ #
    def update_and_get_mi(self, signal_sign: int, mu: float) -> float:
        """Update counts from last prediction vs current move; return MI.

        Steps:
          1. If we have a previous observation, compute current move sign,
             decay all counts, then increment the (prev_signal, move) cell.
          2. Store current (signal_sign, mu) for the next call.
          3. Compute and return MI from the updated table.
        """
        if self.prev_signal_sign is not None and self.prev_mu is not None:
            delta = mu - self.prev_mu
            if abs(delta) > _EPS:
                move_sign = 1 if delta > 0.0 else 0
                # EMA decay all cells
                for r in range(2):
                    for c in range(2):
                        self.counts[r][c] *= _EMA_DECAY
                # Increment the correct cell
                self.counts[self.prev_signal_sign][move_sign] += 1.0
                self.n_obs += 1

        # Store current state for next step (strictly no lookahead)
        self.prev_signal_sign = signal_sign
        self.prev_mu = mu

        if self.n_obs < _MIN_OBS:
            return 0.0
        return self._compute_mi()

    # ------------------------------------------------------------------ #
    def _compute_mi(self) -> float:
        """MI = Σ p(x,y) · log( p(x,y) / (p(x)·p(y)) )."""
        total = sum(self.counts[r][c] for r in range(2) for c in range(2))
        if total < _EPS:
            return 0.0

        mi = 0.0
        for r in range(2):
            p_r = (self.counts[r][0] + self.counts[r][1]) / total
            for c in range(2):
                p_c = (self.counts[0][c] + self.counts[1][c]) / total
                p_rc = self.counts[r][c] / total
                if p_rc > _EPS and p_r > _EPS and p_c > _EPS:
                    mi += p_rc * math.log(p_rc / (p_r * p_c))
        return max(0.0, mi)


class MutualInformationStrategy:
    """Per-symbol online mutual-information gated strategy."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        std = max(state.fwd_return_std, _EPS)

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        sym = self._get_state(state.symbol)
        sharpe = mu / (std + _EPS)
        signal_sign = 1 if sharpe > 0.0 else 0

        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            # Update state so counts stay live, but emit flat signal
            sym.update_and_get_mi(signal_sign, mu)
            return SignalSet(
                ts=state.ts, symbol=state.symbol,
                momentum=0.0, expected_return=mu,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id, confidence=0.0,
            )

        # Update MI table (prev signal sign vs current observed move)
        mi = sym.update_and_get_mi(signal_sign, mu)

        # MI gate: clamp(MI / MI_max, 0, 1)
        mi_gate = max(0.0, min(1.0, mi / max(_MI_MAX, _EPS)))

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

    # ------------------------------------------------------------------ #
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
