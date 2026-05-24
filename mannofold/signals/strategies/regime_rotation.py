"""Regime-rotation strategy: directional stance driven by regime expected drift.

Goes long in bullish neighbourhoods (positive fwd_return_mean), short in bearish
ones, sizing by magnitude via tanh. Hysteresis per-symbol prevents whipsaw on
regime churn: only enter when conviction clears ENTRY_THRESH, hold until it
falls below EXIT_THRESH.
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

NAME = "regime_rotation"
DESCRIPTION = "Regime-drift directional sizing with per-symbol hysteresis to avoid whipsaw."

# Tunable knobs
_GAIN = 3.0           # amplifier inside tanh( gain * mean / std )
_ANOMALY_THRESH = 0.6 # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04     # collapse |weight| below this to 0
_ENTRY_THRESH = 0.10  # minimum |weight| to open / flip a position
_EXIT_THRESH = 0.05   # hold until |weight| drops below this


class RegimeRotationStrategy:
    """Directional regime-rotation with hysteresis."""

    def __init__(self) -> None:
        # per-symbol: current signed position (+1, -1, or 0)
        self._stance: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Sharpe-normalised drift: the core regime signal
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,           # raw Sharpe; target() applies tanh(gain * .)
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Proposed raw weight before hysteresis:
        # target_weight = tanh(gain * fwd_return_mean / (fwd_return_std + 1e-9)) * confidence
        # signals.momentum already holds the Sharpe (mean/std), so:
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: treat tiny weights as zero
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        current_stance = self._stance.get(sym, 0)
        desired_sign = 1 if raw > 0 else (-1 if raw < 0 else 0)

        if current_stance == 0:
            # No position — only enter if conviction clears entry threshold
            if abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            # Holding a position — flip only if new side clears entry threshold
            if desired_sign != 0 and desired_sign != current_stance and abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            # Exit if conviction decays below exit threshold
            elif abs(raw) < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
            # Otherwise hold current stance but scale by current raw magnitude
            else:
                pass  # keep current_stance; weight computed below

        # Emit weight aligned to current stance
        weight = abs(raw) * self._stance[sym]
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh RegimeRotationStrategy instance."""
    return RegimeRotationStrategy()
