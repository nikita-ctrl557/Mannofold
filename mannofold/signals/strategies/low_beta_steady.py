"""Low-beta steady strategy.

Prefer LOW-dispersion, HIGH-density manifold states and shrink exposure as
dispersion or anomaly rise, so the book has low sensitivity to volatility
spikes (low "beta") and steady month-over-month consistency.

Weight formula:
    sharpe  = fwd_return_mean / (fwd_return_std + eps)
    base    = sign(sharpe) * tanh(gain * |tanh(sharpe)|)
    disp    = 1 / (1 + k * fwd_return_std)          # dispersion shrink
    conf    = regime_prob * (1 - anomaly_score)
    weight  = base * disp * density_gate * conf

Hard gates: ANOMALY_REGIME or anomaly > 0.6 -> flat.
Dead-band: |weight| < 0.04 -> flat.
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

NAME = "low_beta_steady"
DESCRIPTION = (
    "Low-beta steady: size = sign(sharpe)*tanh(gain*|tanh(sharpe)|) scaled by "
    "inverse-dispersion and density gate, gated by regime confidence and anomaly."
)

_EPS = 1e-9
# Gain inside outer tanh — controls saturation speed.
_GAIN = 60.0
# Dispersion-shrink steepness; higher k -> harder penalty on large std.
_K_DISP = 40.0
# Logistic density gate parameters (same convention as density_gated.py).
_DENSITY_MID = 1.0
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0
# Anomaly gate threshold.
_ANOMALY_GATE = 0.6
# Dead-band.
_DEADBAND = 0.04


def _density_gate(density: float) -> float:
    """Smooth sigmoid gate: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class LowBetaSteadyStrategy:
    """Minimise volatility-spike sensitivity via dispersion and density gating."""

    def __init__(
        self,
        gain: float = _GAIN,
        k_disp: float = _K_DISP,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._gain = gain
        self._k_disp = k_disp
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        tanh_sharpe = math.tanh(sharpe)
        # base: sign(sharpe) * tanh(gain * |tanh(sharpe)|)
        base = math.copysign(1.0, sharpe) * math.tanh(self._gain * abs(tanh_sharpe))

        # Dispersion shrink: 1 / (1 + k * std)
        disp = 1.0 / (1.0 + self._k_disp * state.fwd_return_std)

        # Density gate: favour high-density (typical) regions.
        gate = _density_gate(state.density)

        # Confidence: regime stability * (1 - anomaly)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Combined momentum signal stores base*disp; gate and confidence folded in.
        momentum = base * disp * gate

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or elevated anomaly -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trading near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return LowBetaSteadyStrategy()
