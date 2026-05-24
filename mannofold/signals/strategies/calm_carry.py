"""Calm-carry strategy.

Takes carry/drift exposure ONLY when conditions are calm and favorable:
low dispersion (small fwd_return_std), high manifold density, confirmed
positive persistent drift, and high confidence.  In choppy, high-vol,
or uncertain states the strategy stays flat (weight < dead-band).

Two EMA layers stabilise the signal:
  1. fwd_return_mean EMA (per symbol): captures PERSISTENT regime drift,
     smoothing out noisy single-bar neighbourhood estimates.  Only trade
     when this EMA is positive (confirmed positive carry).
  2. Weight EMA (per symbol): smooths the target weight across bars,
     reducing turnover and allowing carry to compound within a regime.

Signal:
    ema_mean  = alpha_m * fwd_return_mean + (1-alpha_m) * prev_ema_mean
    sharpe    = ema_mean / (fwd_return_std + eps)       [clamped ±10]
    base      = tanh(gain * tanh(sharpe))
    calm_gate = sigma(density_scale*(density - density_mid))
              * sigma(-std_scale*(fwd_return_std - std_mid))
              ~ 1 when density HIGH AND fwd_return_std LOW
    confidence = regime_prob * (1 - anomaly_score)
    raw_weight = base * calm_gate * confidence
    weight     = alpha_w * raw_weight + (1-alpha_w) * prev_weight  [EMA]

Gates:
    flat on ANOMALY_REGIME or anomaly_score > 0.6
    flat when ema_mean <= 0 (only harvest confirmed positive drift/carry)
    dead-band |weight| < 0.04 -> 0
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
    "regimes with confirmed persistent positive drift (EMA of fwd_return_mean). "
    "Flat on anomaly or non-positive drift; weight-EMA smoothing reduces turnover "
    "and compounds carry gains across consecutive calm bars."
)

_EPS = 1e-9
# Tanh gain applied to Sharpe proxy — moderate to stay selective.
_GAIN = 2.5
# EMA alpha for smoothing fwd_return_mean per symbol (lower = more smoothing).
_EMA_ALPHA_MEAN = 0.15
# EMA alpha for smoothing target weight per symbol (reduces turnover).
_EMA_ALPHA_WEIGHT = 0.25
# Anomaly score threshold above which we go flat.
_ANOMALY_GATE = 0.6
# Dead-band: weights below this are zeroed to avoid noise trading.
_DEADBAND = 0.04
# Density gate: logistic sigmoid.  Gate -> 1 when density > _DENSITY_MID.
# Range ~0.18..0.77; mid 0.35 with scale 8 lets most typical bars pass.
_DENSITY_MID = 0.35
_DENSITY_SCALE = 8.0
_DENSITY_CLAMP = 50.0
# Vol gate: inverse sigmoid.  Gate -> 1 when fwd_return_std < _STD_MID.
# Median std ~0.018; mid 0.021 softly penalises high-vol bars.
_STD_MID = 0.021
_STD_SCALE = 120.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid in (0, 1)."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _calm_gate(density: float, fwd_return_std: float) -> float:
    """Joint calmness gate in [0,1]: ~1 when density HIGH and std LOW."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    density_g = _sigmoid(_DENSITY_SCALE * (d - _DENSITY_MID))
    std_g = _sigmoid(-_STD_SCALE * (fwd_return_std - _STD_MID))
    return density_g * std_g


class CalmCarryStrategy:
    """Carry exposure gated on calm conditions, persistent drift, and confidence."""

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_mean (for persistent drift signal).
        self._ema_mean: dict[str, float] = {}
        # Per-symbol EMA of target weight (for turnover reduction).
        self._ema_weight: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Update per-symbol EMA of fwd_return_mean (no lookahead).
        prev_ema = self._ema_mean.get(sym, state.fwd_return_mean)
        ema = _EMA_ALPHA_MEAN * state.fwd_return_mean + (1.0 - _EMA_ALPHA_MEAN) * prev_ema
        self._ema_mean[sym] = ema

        # Sharpe proxy using EMA-smoothed drift.
        sharpe = ema / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        gate = _calm_gate(state.density, state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=math.tanh(sharpe),      # tanh(sharpe) — used in target()
            expected_return=ema,              # EMA drift — checked for positivity
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,     # calm_gate fused into confidence
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly -> flat; decay weight EMA.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            self._ema_weight[sym] = (
                (1.0 - _EMA_ALPHA_WEIGHT) * self._ema_weight.get(sym, 0.0)
            )
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Only harvest confirmed positive carry; decay weight EMA when drifting down.
        if signals.expected_return <= 0.0:
            self._ema_weight[sym] = (
                (1.0 - _EMA_ALPHA_WEIGHT) * self._ema_weight.get(sym, 0.0)
            )
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Raw weight: tanh(gain*tanh(sharpe)) * calm_gate_fused_confidence.
        base = math.tanh(_GAIN * signals.momentum)
        raw_weight = base * signals.confidence

        # Smooth the weight to reduce turnover and compound calm carry.
        prev_w = self._ema_weight.get(sym, 0.0)
        smoothed = _EMA_ALPHA_WEIGHT * raw_weight + (1.0 - _EMA_ALPHA_WEIGHT) * prev_w
        self._ema_weight[sym] = smoothed

        # Dead-band: collapse small weights to flat.
        weight = 0.0 if abs(smoothed) < _DEADBAND else smoothed
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return CalmCarryStrategy()
