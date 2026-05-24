"""Calm-carry strategy.

Takes carry/drift exposure ONLY when conditions are calm and favorable:
low dispersion (small fwd_return_std), high manifold density, positive
persistent drift (EMA of fwd_return_mean), and high confidence.  In
choppy, high-vol, or uncertain states the strategy stays flat.

The drift signal uses an EMA-smoothed fwd_return_mean (per symbol) rather
than the raw single-bar value; this captures PERSISTENT regime drift and
avoids reacting to isolated noisy neighbourhood estimates.

Signal:
    ema_mean    updated online: alpha*fwd_return_mean + (1-alpha)*prev_ema
    sharpe      = ema_mean / (fwd_return_std + eps)
    base        = tanh(gain * tanh(sharpe))
    calm_gate   = sigma(density_scale*(density - density_mid))
                * sigma(-std_scale*(fwd_return_std - std_mid))
                ~ 1 only when density HIGH and fwd_return_std LOW
    confidence  = regime_prob * (1 - anomaly_score)
    weight      = base * calm_gate * confidence

Gates:
    flat on ANOMALY_REGIME or anomaly_score > 0.6
    flat when ema_mean <= 0 (only harvest confirmed positive carry)
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
    "regimes with confirmed positive drift (EMA of fwd_return_mean).  Flat "
    "whenever volatility is elevated, density is low, EMA drift is non-positive, "
    "confidence is weak, or an anomalous regime is detected."
)

_EPS = 1e-9
# tanh gain applied to Sharpe proxy.
_GAIN = 2.5
# EMA alpha for smoothing fwd_return_mean.  Lower = more smoothing.
_EMA_ALPHA = 0.10
# Anomaly score threshold above which we go flat.
_ANOMALY_GATE = 0.6
# Dead-band: negligible weights collapsed to zero to avoid noise trading.
_DEADBAND = 0.04
# Density gate: logistic sigmoid.  Gate -> 1 when density > _DENSITY_MID.
_DENSITY_MID = 0.48
_DENSITY_SCALE = 10.0
_DENSITY_CLAMP = 50.0
# Vol gate: inverse sigmoid.  Gate -> 1 when fwd_return_std < _STD_MID.
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
    """Carry exposure gated on calm market conditions, persistent drift, and confidence."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean; initialised lazily on first tick.
        self._ema_mean: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # --- Update per-symbol EMA of fwd_return_mean (no lookahead) ---
        prev_ema = self._ema_mean.get(sym, state.fwd_return_mean)
        ema = _EMA_ALPHA * state.fwd_return_mean + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema_mean[sym] = ema

        # Sharpe proxy using EMA-smoothed drift (persistent carry signal).
        sharpe = ema / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        gate = _calm_gate(state.density, state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Store ema_mean in expected_return for the gate check in target().
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=math.tanh(sharpe),     # tanh(sharpe) passed to target()
            expected_return=ema,             # EMA drift — used for direction gate
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,   # calm_gate fused into confidence
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Only harvest when EMA drift is confirmed positive (carry direction).
        if signals.expected_return <= 0.0:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Base signal: tanh(gain * tanh(sharpe)).  signals.momentum = tanh(sharpe).
        base = math.tanh(_GAIN * signals.momentum)

        # weight = base * calm_gate_fused_confidence.
        weight = base * signals.confidence

        # Dead-band: collapse small weights to flat to avoid noise trading.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return CalmCarryStrategy()
