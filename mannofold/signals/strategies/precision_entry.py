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

# ── Gate thresholds (calibrated: density mid=0.4, regime_prob proxy via conf) ─
# All must pass simultaneously; the conjunction keeps entry rate low.
_ANOMALY_MAX = 0.35          # reject noisy / off-manifold states
_DENSITY_GATE_MIN = 0.50     # density sigmoid must clear mid-point
_SHARPE_ABS_MIN = 0.30       # |sharpe| must show clear directional edge
# Confidence floor proxies regime_prob > 0.65 given anomaly < 0.35:
# conf = regime_prob*(1-anomaly_score); at rp=0.65, anom=0.35 -> conf=0.4225
_CONFIDENCE_MIN = 0.4225     # ~= 0.65 * (1 - 0.35)

# ── Sizing parameters ────────────────────────────────────────────────────────
_GAIN = 2.5                  # amplifier inside tanh(gain * tanh(sharpe))
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

        # Recover sharpe from tanh(sharpe) (clamp to avoid atanh instability)
        sharpe_proxy = math.atanh(max(-1.0 + _EPS, min(1.0 - _EPS, tanh_sharpe)))

        # ── ALL quality gates must pass simultaneously ──
        if dg < _DENSITY_GATE_MIN:        # require high-density (typical) region
            return zero
        if anomaly > _ANOMALY_MAX:        # reject high-anomaly states
            return zero
        if confidence < _CONFIDENCE_MIN:  # proxies regime_prob > ~0.65 when anom < 0.35
            return zero
        if abs(sharpe_proxy) < _SHARPE_ABS_MIN:  # require clear directional drift
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
