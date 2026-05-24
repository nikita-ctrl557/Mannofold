"""Precision-entry strategy: only take RARE, high-probability setups.

All quality gates must pass simultaneously before entering. The high bar
means few trades, but each one carries positive expectancy by construction.
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

NAME = "precision_entry"
DESCRIPTION = (
    "Flat by default; enter only when regime_prob, density, anomaly, AND "
    "directional Sharpe all clear strict thresholds simultaneously. "
    "Rare but high-hit-rate trades with positive expectancy."
)

_EPS = 1e-9

# ── Gate thresholds (calibrated to synthetic data distributions) ─────────────
_REGIME_PROB_MIN = 0.60      # regime must be reasonably stable (median ~0.73)
_ANOMALY_MAX = 0.35          # reject noisy / off-manifold states (median ~0.23)
_DENSITY_GATE_MIN = 0.50     # density sigmoid must be in typical region
_SHARPE_ABS_MIN = 0.25       # |sharpe| must show a clear directional edge (median ~0.28)
_CONFIDENCE_MIN = 0.20       # joint confidence floor (regime_prob*(1-anomaly)*dg)

# ── Sizing parameters ────────────────────────────────────────────────────────
_GAIN = 3.0                  # amplifier inside tanh(gain * tanh(sharpe))
_DEAD_BAND = 0.04            # collapse tiny weights to flat

# ── Density sigmoid parameters (mid=0.4 so gate is ~0.5 at typical density) ──
_DENSITY_MID = 0.4
_DENSITY_SCALE = 3.0
_DENSITY_CLAMP = 50.0


def _density_gate(density: float) -> float:
    """Smooth [0,1] gate: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class PrecisionEntryStrategy:
    """All-gates-must-pass strategy optimising for win rate over trade frequency."""

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        dg = _density_gate(state.density)
        # Confidence = regime stability * non-anomalousness * density quality
        confidence = max(
            0.0,
            min(1.0, state.regime_prob * (1.0 - state.anomaly_score) * dg),
        )
        # Pack density gate value into momentum field for use in target()
        # expected_return carries tanh(sharpe) for direction + magnitude
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=dg,                         # density gate value [0,1]
            expected_return=math.tanh(sharpe),   # tanh(sharpe) ~ direction
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gate: anomalous regime -> always flat
        if signals.regime_id == ANOMALY_REGIME:
            return zero

        dg = signals.momentum                    # density gate packed in signals()
        tanh_sharpe = signals.expected_return    # tanh(sharpe)
        confidence = signals.confidence
        anomaly = signals.anomaly

        # Recover sharpe proxy from tanh(sharpe)
        sharpe_proxy = math.atanh(max(-1 + _EPS, min(1 - _EPS, tanh_sharpe)))

        # ── All quality gates must pass simultaneously ──
        if dg < _DENSITY_GATE_MIN:
            return zero
        if anomaly > _ANOMALY_MAX:
            return zero
        if confidence < _CONFIDENCE_MIN:
            return zero
        if abs(sharpe_proxy) < _SHARPE_ABS_MIN:
            return zero

        # All gates cleared — weight = tanh(gain*tanh(sharpe)) * confidence
        weight = math.tanh(_GAIN * tanh_sharpe) * confidence

        # Dead-band: collapse negligible weights to flat
        if abs(weight) < _DEAD_BAND:
            return zero

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh PrecisionEntryStrategy with default parameters."""
    return PrecisionEntryStrategy()
