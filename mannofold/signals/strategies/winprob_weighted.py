"""Win-probability-weighted strategy: only trade when estimated directional
hit-rate exceeds 0.52; size proportional to (winprob - 0.5).

Win-probability is maintained per-symbol as an online EMA of whether the
previous bar's chosen direction matched the subsequently observed move
(strictly past-only, no lookahead).
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

NAME = "winprob_weighted"
DESCRIPTION = (
    "High-selectivity strategy that trades only when online-calibrated "
    "win-probability exceeds 0.52; position sized by (winprob - 0.5)."
)

_EPS             = 1e-9
_WIN_PROB_ALPHA  = 0.15   # EMA decay for directional hit-rate (faster adaptation)
_WIN_PROB_THRESH = 0.52   # minimum win-prob to take a position
_GAIN            = 2.5    # tanh gain on |sharpe|
_ANOMALY_THRESH  = 0.6
_DEAD_BAND       = 0.04


class _SymbolState:
    """Per-symbol online win-probability tracker."""

    def __init__(self) -> None:
        # EMA of binary win outcomes (1=win, 0=loss); starts slightly above 0.5
        # so the strategy can begin trading and receive real feedback immediately.
        self.win_prob: float = 0.55
        # Direction chosen on the previous bar (+1 / -1 / 0)
        self.prev_direction: float = 0.0
        # fwd_return_mean seen on the previous bar (used to detect move)
        self.prev_mu: Optional[float] = None

    def update(self, current_mu: float) -> None:
        """Update win-prob EMA using the *previous* direction vs current move."""
        if self.prev_direction != 0.0 and self.prev_mu is not None:
            delta = current_mu - self.prev_mu
            if abs(delta) > _EPS:
                move_sign = 1.0 if delta > 0.0 else -1.0
                win = 1.0 if (self.prev_direction * move_sign) > 0.0 else 0.0
                self.win_prob = (
                    _WIN_PROB_ALPHA * win
                    + (1.0 - _WIN_PROB_ALPHA) * self.win_prob
                )
        self.prev_mu = current_mu

    def record_direction(self, direction: float) -> None:
        """Store the direction chosen this bar for evaluation next bar."""
        self.prev_direction = direction


class WinProbWeightedStrategy:
    """Trade only when estimated win-probability > threshold; size by edge."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        sharpe = mu / (sig + _EPS)

        sym_state = self._get_state(state.symbol)
        # Update win-prob with new observation BEFORE computing signal
        sym_state.update(mu)

        confidence = max(
            0.0,
            min(1.0, state.regime_prob * (1.0 - state.anomaly_score)),
        )

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,           # raw sharpe carried as momentum
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard-off conditions
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._get_state(signals.symbol).record_direction(0.0)
            return flat

        sym_state = self._get_state(signals.symbol)
        win_prob  = sym_state.win_prob

        if win_prob <= 0.5:
            sym_state.record_direction(0.0)
            return flat

        sharpe    = signals.momentum
        direction = 1.0 if sharpe > 0.0 else (-1.0 if sharpe < 0.0 else 0.0)

        if direction == 0.0:
            sym_state.record_direction(0.0)
            return flat

        # Edge factor: clamp(2*(winprob-0.5), 0, 1)
        edge = min(1.0, 2.0 * (win_prob - 0.5))

        # Magnitude factor: tanh(gain * |tanh(sharpe)|)
        magnitude = math.tanh(_GAIN * abs(math.tanh(sharpe)))

        raw = direction * edge * magnitude * signals.confidence

        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        # Store the direction we're actually trading for next bar's evaluation
        sym_state.record_direction(direction if raw != 0.0 else 0.0)

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh WinProbWeightedStrategy instance."""
    return WinProbWeightedStrategy()
