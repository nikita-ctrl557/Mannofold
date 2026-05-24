"""Lyapunov-stability manifold strategy.

Estimates a local Lyapunov-like exponent λ from the manifold trajectory —
the log growth rate of the embedding velocity magnitude vs its recent EMA:

    λ ≈ log(|velocity_now| / (ema_|velocity| + eps))

λ < 0 → trajectory contracting → stable/predictable → trust the drift, size up.
λ > 0 → trajectory expanding / chaotic → unpredictable → cut to flat.

target_weight = tanh(gain · tanh(sharpe)) · clamp(exp(-max(0, λ)), 0, 1) · confidence
confidence    = regime_prob · (1 - anomaly_score)
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

NAME = "lyapunov_stability"
DESCRIPTION = (
    "Sizes positions by a local Lyapunov exponent estimated from manifold velocity: "
    "contracts on stable (λ<0) trajectories, flattens on chaotic (λ>0) ones."
)

_EPS = 1e-9
_GAIN = 60.0          # tanh gain mapping Sharpe → weight direction
_EMA_ALPHA = 0.1      # EMA smoothing for velocity magnitude (per symbol)
_ANOMALY_GATE = 0.6   # anomaly_score above this → flat
_DEADBAND = 0.04      # |weight| below this collapses to 0


class _LyapunovSignalSet(SignalSet):
    """SignalSet carrying the Lyapunov exponent for use in target()."""

    lyapunov: float = 0.0

    model_config = {"extra": "allow"}


class LyapunovStabilityStrategy:
    """Manifold strategy that gates conviction on the local Lyapunov exponent."""

    def __init__(
        self,
        gain: float = _GAIN,
        ema_alpha: float = _EMA_ALPHA,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._ema_alpha = ema_alpha
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        # Per-symbol EMA of velocity magnitude (no lookahead).
        self._ema_speed: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Current velocity magnitude.
        speed_now = math.sqrt(sum(v * v for v in state.velocity)) if state.velocity else 0.0

        # Update per-symbol EMA of speed (online, no lookahead).
        sym = state.symbol
        ema = self._ema_speed.get(sym, speed_now)
        ema = self._ema_alpha * speed_now + (1.0 - self._ema_alpha) * ema
        self._ema_speed[sym] = ema

        # Local Lyapunov exponent estimate.
        lyapunov = math.log(speed_now / (ema + _EPS)) if speed_now > 0.0 else 0.0

        return _LyapunovSignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            lyapunov=lyapunov,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard-off on anomalous regime or high anomaly score.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        lyapunov = getattr(signals, "lyapunov", 0.0)

        # Stability scaler: exp(-max(0, λ)) ∈ (0, 1].
        # λ ≤ 0 → scaler = 1 (full trust); λ > 0 → scaler decays toward 0.
        stability = math.exp(-max(0.0, lyapunov))
        stability = max(0.0, min(1.0, stability))

        # signals.momentum already holds tanh(sharpe).
        w = math.tanh(self._gain * signals.momentum) * stability * signals.confidence

        # Dead-band.
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a LyapunovStabilityStrategy with default parameters."""
    return LyapunovStabilityStrategy()
