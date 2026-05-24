"""Kalman Trend strategy: scalar Kalman filter on latent drift.

Models the per-symbol latent drift as a hidden state estimated by a
scalar Kalman filter. The measurement is fwd_return_mean; measurement
noise R = fwd_return_std^2 (noisy observations receive less weight).
The filtered estimate x̂ is denoised drift; position is sized by
tanh(gain * x̂ / sqrt(R)) weighted by Kalman confidence and regime
confidence. Strictly causal — no lookahead.
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

NAME = "kalman_trend"
DESCRIPTION = (
    "Scalar Kalman filter on latent drift: denoises fwd_return_mean via "
    "optimal estimation; sizes positions by filtered drift scaled by "
    "Kalman gain (confidence) and regime probability."
)

# Tunable knobs
_PROCESS_NOISE_Q = 1e-5   # state transition noise (small = slow-moving drift)
_GAIN_SCALE = 3.0         # amplifier inside tanh for position sizing
_ANOMALY_THRESH = 0.6     # anomaly_score above this -> flat
_DEAD_BAND = 0.04         # collapse |weight| below this to 0


class _KalmanState:
    """Per-symbol scalar Kalman filter state."""

    __slots__ = ("x_hat", "P")

    def __init__(self) -> None:
        self.x_hat: float = 0.0   # state estimate (latent drift)
        self.P: float = 1.0       # estimate variance


class KalmanTrendStrategy:
    """Kalman-filter trend strategy with per-symbol state."""

    def __init__(self) -> None:
        self._states: dict[str, _KalmanState] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Retrieve or initialise per-symbol Kalman state
        if sym not in self._states:
            self._states[sym] = _KalmanState()
        ks = self._states[sym]

        # Measurement noise: R = fwd_return_std^2 (higher std -> less trust)
        R = state.fwd_return_std ** 2 + 1e-9

        # --- Predict step ---
        # State mean unchanged (random-walk drift model); variance grows by Q
        P_pred = ks.P + _PROCESS_NOISE_Q

        # --- Update step ---
        K = P_pred / (P_pred + R)                          # Kalman gain
        innovation = state.fwd_return_mean - ks.x_hat      # measurement residual
        x_hat_new = ks.x_hat + K * innovation              # updated state estimate
        P_new = (1.0 - K) * P_pred                         # updated variance

        # Save updated state for the NEXT call (current bar used current obs)
        # We compute signals from the POST-update estimate (still causal:
        # the filter only uses data up to and including the current bar).
        ks.x_hat = x_hat_new
        ks.P = P_new

        # Confidence: Kalman gain encodes how much new info mattered;
        # high K (uncertain prior) -> more adaptive, combine with regime quality
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Encode filtered drift as momentum signal
        # (signed, scaled by Kalman gain so uncertain estimates are smaller)
        sqrt_R = math.sqrt(R)
        normalised_drift = x_hat_new / (sqrt_R + 1e-9)
        momentum = math.tanh(_GAIN_SCALE * normalised_drift) * K

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=x_hat_new,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Target weight: filtered direction * Kalman confidence * regime confidence
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh KalmanTrendStrategy instance."""
    return KalmanTrendStrategy()
