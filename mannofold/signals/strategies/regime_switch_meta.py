"""Regime-switch meta strategy: blend momentum-follow and mean-reversion based on dispersion.

Different engines win in different regimes:
  - Momentum-style wins in clean trending / low-dispersion states.
  - Mean-reversion wins in choppy / high-dispersion states.

This meta engine exploits that finding by blending the two approaches smoothly
via a logistic function of a running dispersion measure, keeping per-symbol EMA
state (deterministic, no lookahead).
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

NAME = "regime_switch_meta"
DESCRIPTION = "Blend momentum-follow vs mean-reversion via logistic of dispersion EMA; flat on anomaly."

# Tunable knobs
_GAIN = 2.5           # amplifier inside the outer tanh for both arms
_ANOMALY_THRESH = 0.6 # anomaly_score above this -> flat
_DEAD_BAND = 0.04     # collapse |weight| below this to 0
_EPS = 1e-9

# EMA decay for per-symbol running dispersion (fwd_return_std)
_DISP_ALPHA = 0.05    # slow EMA: ~1/alpha = 20 bars half-life

# Logistic parameters for dispersion gate:
# blend_mr = sigmoid(scale * (disp_ratio - mid))
# disp_ratio = current_std / ema_std (relative measure)
# When disp_ratio > mid -> lean mean-reversion; when < mid -> lean momentum
_DISP_MID = 1.2       # ratio threshold: above this means "choppy"
_DISP_SCALE = 4.0     # steepness of the logistic transition

# Density logistic gate: low density -> reduce conviction
_DENSITY_MID = 1.0
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0


def _logistic(x: float, scale: float, mid: float) -> float:
    """Smooth sigmoid gate, output in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-scale * (x - mid)))


def _density_gate(density: float) -> float:
    """Map density to a [0,1] conviction gate; low density -> ~0."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return _logistic(d, _DENSITY_SCALE, _DENSITY_MID)


class RegimeSwitchMetaStrategy:
    """Meta engine that adaptively blends momentum-follow and mean-reversion.

    Core idea:
      sharpe  = fwd_return_mean / (fwd_return_std + eps)
      mom_arm = +tanh(gain * tanh(sharpe))   # follow the drift
      rev_arm = -tanh(gain * tanh(sharpe))   # fade the drift

      disp_ratio = fwd_return_std / ema_std  (running per-symbol EMA of std)
      blend_mr   = sigmoid(disp_scale * (disp_ratio - disp_mid))  in (0,1)
      blend_mom  = 1 - blend_mr

      raw_weight = blend_mom * mom_arm + blend_mr * rev_arm
                 = (1 - 2*blend_mr) * tanh(gain * tanh(sharpe))

    Gated by: confidence = regime_prob * (1-anomaly_score) * density_gate.
    Flat on ANOMALY_REGIME or anomaly > _ANOMALY_THRESH.
    Dead-band |w| < _DEAD_BAND -> 0.
    """

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_std (dispersion)
        self._disp_ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        std = state.fwd_return_std

        # Update per-symbol EMA of dispersion
        if sym not in self._disp_ema:
            self._disp_ema[sym] = max(std, _EPS)
        else:
            self._disp_ema[sym] = (
                _DISP_ALPHA * std + (1.0 - _DISP_ALPHA) * self._disp_ema[sym]
            )

        ema_std = max(self._disp_ema[sym], _EPS)

        # Dispersion ratio: >1 means current volatility is above its running mean
        disp_ratio = std / ema_std

        # Compute momentum signal (tanh of Sharpe)
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)

        # Confidence fuses regime certainty, anomaly, and density typicality
        confidence = state.regime_prob * (1.0 - state.anomaly_score) * _density_gate(state.density)
        confidence = max(0.0, min(1.0, confidence))

        # Encode dispersion blend factor into the SignalSet confidence field
        # We pass disp_ratio via expected_return scratch storage isn't available,
        # so we compute final weight here and store as momentum for target() to use.
        # Blend: high disp_ratio -> mean-reversion arm dominates
        blend_mr = _logistic(disp_ratio, _DISP_SCALE, _DISP_MID)
        # Signed direction: +1 momentum, -1 mean-reversion, smooth blend
        direction_scale = 1.0 - 2.0 * blend_mr  # in (-1, 1)
        blended_momentum = direction_scale * momentum  # signed momentum with blend

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=blended_momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # momentum already encodes the blend: tanh(gain * blended_momentum) * confidence
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh RegimeSwitchMetaStrategy instance."""
    return RegimeSwitchMetaStrategy()
