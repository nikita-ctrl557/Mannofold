"""Defensive-compounder strategy.

Month-over-month CONSISTENCY: small steady gains that compound.  Always takes
SMALL positions (|weight| capped at 0.4) aligned with high-confidence drift in
calm/typical states.  Half-Kelly-ish Sharpe sizing capped low, density gate,
and a confidence gate (regime_prob * (1 - anomaly_score)) keep the strategy
selective.  The aim is a high fraction of small positive periods that compound
rather than large bets that can blow up.

Signal:
    sharpe       = fwd_return_mean / (fwd_return_std + eps)   (clipped ±10)
    base         = clamp(tanh(gain * tanh(sharpe)), -0.4, 0.4)
    density_gate = sigmoid(density_scale * (density - density_mid))
    confidence   = regime_prob * (1 - anomaly_score)
    weight       = base * density_gate * confidence

Gates:
    flat on ANOMALY_REGIME or anomaly_score > 0.6
    dead-band |w| < 0.04 -> 0
    no lookahead; build() takes no args
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

NAME = "defensive_compounder"
DESCRIPTION = (
    "Month-over-month consistency compounder: always takes small positions "
    "(|weight| <= 0.4) in high-confidence drift aligned with calm, high-density "
    "regimes.  Half-Kelly-ish Sharpe sizing with hard weight cap, density gate, "
    "and regime-confidence attenuation maximize win-rate over magnitude."
)

_EPS = 1e-9
# tanh gain applied to tanh(sharpe) — moderate to keep sizing conservative.
_GAIN = 2.0
# Hard cap on |weight| — defensive: never more than 40% exposure.
_WEIGHT_CAP = 0.4
# Anomaly gate: go flat when anomaly score exceeds this threshold.
_ANOMALY_GATE = 0.6
# Dead-band: collapse negligible weights to zero to avoid noise trading.
_DEADBAND = 0.04
# Density gate: logistic sigmoid. Gate -> 1 when density > _DENSITY_MID.
# Density range ~0.18..0.77; mid at 0.45 passes the denser/calmer half.
_DENSITY_MID = 0.45
_DENSITY_SCALE = 10.0
_DENSITY_CLAMP = 50.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _density_gate(density: float) -> float:
    """Smooth gate in [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return _sigmoid(_DENSITY_SCALE * (d - _DENSITY_MID))


class DefensiveCompounderStrategy:
    """Small-position compounder: high win-rate over magnitude."""

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        gate = _density_gate(state.density)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),         # tanh(sharpe) for target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,        # density gate fused in
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # base = clamp(tanh(gain * tanh(sharpe)), -cap, cap)
        # signals.momentum == tanh(sharpe) from signals().
        base = math.tanh(_GAIN * signals.momentum)
        base = max(-_WEIGHT_CAP, min(_WEIGHT_CAP, base))

        # weight = base * density_gate_fused_confidence
        weight = base * signals.confidence

        # Dead-band: collapse small weights to flat.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        # Final clamp to weight cap (belt-and-suspenders).
        weight = max(-_WEIGHT_CAP, min(_WEIGHT_CAP, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return DefensiveCompounderStrategy()
