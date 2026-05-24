"""Calm-carry strategy.

Takes carry/drift exposure ONLY when conditions are calm and favorable:
low dispersion (small fwd_return_std), high manifold density, and high
confidence (regime_prob * (1 - anomaly_score)).  In choppy, high-vol, or
uncertain states the strategy stays flat.  The signal is BIDIRECTIONAL —
long when neighbourhood drift is positive, short when negative — so the
strategy captures both sides of calm carry while staying flat in noise.

Signal:
    sharpe      = fwd_return_mean / (fwd_return_std + eps)
    base        = tanh(gain * tanh(sharpe))
    calm_gate   = sigma(density_scale*(density - density_mid))
                * sigma(-std_scale*(fwd_return_std - std_mid))
                ~ 1 only when density HIGH and fwd_return_std LOW
    confidence  = regime_prob * (1 - anomaly_score)
    weight      = base * calm_gate * confidence

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

NAME = "calm_carry"
DESCRIPTION = (
    "Collects steady small carry gains in calm, high-density, low-dispersion "
    "regimes only.  Bidirectional: long on positive neighbourhood drift, short "
    "on negative drift; flat whenever volatility is elevated, density is low, "
    "confidence is weak, or an anomalous regime is detected."
)

_EPS = 1e-9
# tanh gain applied to Sharpe proxy — kept moderate to stay selective.
_GAIN = 2.5
# Anomaly score threshold above which we go flat.
_ANOMALY_GATE = 0.6
# Dead-band: negligible weights collapsed to zero to avoid noise trading.
_DEADBAND = 0.04
# Density gate: logistic sigmoid.  Gate -> 1 when density > _DENSITY_MID.
# Density range ~0.18..0.77; mid at 0.48 lets calmer/denser half pass.
_DENSITY_MID = 0.48
_DENSITY_SCALE = 10.0
_DENSITY_CLAMP = 50.0
# Vol gate: inverse sigmoid.  Gate -> 1 when fwd_return_std < _STD_MID.
# Median std ~0.018; mid at 0.020 lets lower-vol half pass.
_STD_MID = 0.020
_STD_SCALE = 150.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid in (0, 1)."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _calm_gate(density: float, fwd_return_std: float) -> float:
    """Joint calmness gate: ~1 when density is high AND std is low."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    density_g = _sigmoid(_DENSITY_SCALE * (d - _DENSITY_MID))
    std_g = _sigmoid(-_STD_SCALE * (fwd_return_std - _STD_MID))
    return density_g * std_g


class CalmCarryStrategy:
    """Carry exposure gated on calm market conditions and regime confidence."""

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        gate = _calm_gate(state.density, state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Embed calm_gate into confidence so target() receives the fused value.
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),         # tanh(sharpe) — passed to target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,        # calm_gate fused in
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Reconstruct base: tanh(gain * tanh(sharpe)).
        # signals.momentum == tanh(sharpe) from signals().
        base = math.tanh(_GAIN * signals.momentum)

        # weight = base * calm_gate_fused_confidence.
        weight = base * signals.confidence

        # Dead-band: collapse small weights to flat.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return CalmCarryStrategy()
