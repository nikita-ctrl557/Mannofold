"""Equal-risk blend strategy: three sub-signals each normalized to unit recent vol.

Blends drift, carry, and density-gated mean-reversion, dividing each by its
per-symbol EMA of |sub_signal| so that every signal contributes equal RISK to
the composite. This diversification smooths month-over-month returns.

composite = sum(sub_i / (ema_abs_i + eps)) / 3
weight    = tanh(gain * composite) * confidence
confidence = regime_prob * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > 0.6; dead-band |w| < 0.04 -> 0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import DefaultDict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "equal_risk_blend"
DESCRIPTION = (
    "Equal-risk blend of drift, carry, and density-gated mean-reversion sub-signals "
    "via per-symbol EMA volatility normalisation for steady month-over-month returns."
)

# ------------------------------------------------------------------ #
#  Hyper-parameters                                                   #
# ------------------------------------------------------------------ #
_GAIN            = 2.5    # outer tanh gain on composite
_ANOMALY_THRESH  = 0.6    # anomaly_score above this -> flat
_DEAD_BAND       = 0.04   # |weight| below this -> 0

_EMA_ALPHA       = 0.05   # EMA smoothing factor for |sub_signal| (≈20-bar half-life)
_EMA_INIT        = 0.1    # initial EMA seed (avoids division by near-zero early on)

_EPS             = 1e-9

# Density gate: logistic mid-point and steepness
_DENSITY_MID     = 1.0
_DENSITY_SCALE   = 2.0
_DENSITY_CLAMP   = 50.0

# Carry raw-value clamp before tanh
_CARRY_CLAMP     = 5.0


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _density_gate(density: float) -> float:
    """Smooth gate [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


# ------------------------------------------------------------------ #
#  Strategy class                                                     #
# ------------------------------------------------------------------ #

class EqualRiskBlendStrategy:
    """Blend three sub-signals normalised to equal recent risk contribution."""

    def __init__(self) -> None:
        # Per-symbol EMA of |sub_signal| for each of the three signals
        self._ema_drift:  DefaultDict[str, float] = defaultdict(lambda: _EMA_INIT)
        self._ema_carry:  DefaultDict[str, float] = defaultdict(lambda: _EMA_INIT)
        self._ema_dmrev:  DefaultDict[str, float] = defaultdict(lambda: _EMA_INIT)

    # -------------------------------------------------------------- #

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # --- (a) Drift: Sharpe-like direction ---
        drift = math.tanh(mu / (sig + _EPS))

        # --- (b) Carry: risk-adjusted expected return ---
        carry_raw = mu / (sig ** 2 + _EPS)
        carry = math.tanh(max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw)))

        # --- (c) Density-gated mean-reversion ---
        # Mean-reversion: negative of drift (fade the expected return),
        # gated by how "typical" the current manifold region is.
        gate  = _density_gate(state.density)
        dmrev = -math.tanh(mu / (sig + _EPS)) * gate

        # --- Update per-symbol EMA of |sub_signal| (no lookahead) ---
        self._ema_drift[sym] = (
            _EMA_ALPHA * abs(drift) + (1.0 - _EMA_ALPHA) * self._ema_drift[sym]
        )
        self._ema_carry[sym] = (
            _EMA_ALPHA * abs(carry) + (1.0 - _EMA_ALPHA) * self._ema_carry[sym]
        )
        self._ema_dmrev[sym] = (
            _EMA_ALPHA * abs(dmrev) + (1.0 - _EMA_ALPHA) * self._ema_dmrev[sym]
        )

        # --- Equal-risk normalisation & composite ---
        norm_drift = drift / (self._ema_drift[sym] + _EPS)
        norm_carry = carry / (self._ema_carry[sym] + _EPS)
        norm_dmrev = dmrev / (self._ema_dmrev[sym] + _EPS)
        composite  = (norm_drift + norm_carry + norm_dmrev) / 3.0

        # Confidence: regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=composite,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # -------------------------------------------------------------- #

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


# ------------------------------------------------------------------ #

def build() -> Strategy:
    """Return a fresh EqualRiskBlendStrategy instance."""
    return EqualRiskBlendStrategy()
