"""Wavelet Denoise strategy: multiresolution drift denoising via EMA cascades.

Approximates an à trous (algorithme à trous) wavelet decomposition of the
per-symbol drift signal using a cascade of three EMAs at scales α=0.5, 0.25,
0.1.  Detail coefficients (differences between successive smooth approximations)
are soft-thresholded to suppress noise while preserving signal.  The denoised
drift is then used to size positions with tanh, weighted by a confidence factor
derived from regime probability and anomaly score.  Strictly causal — state is
updated AFTER producing signals from the current bar's posterior.
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

NAME = "wavelet_denoise"
DESCRIPTION = (
    "Multiresolution drift denoising via EMA-cascade soft-thresholding: "
    "approximates an à trous wavelet decomposition, suppresses noise-level "
    "detail coefficients, reconstructs denoised drift, and sizes positions "
    "with tanh scaled by regime × anomaly confidence."
)

# EMA smoothing alphas for three decomposition scales (fast → slow)
_ALPHAS: tuple[float, float, float] = (0.5, 0.25, 0.1)

# Soft-threshold multipliers applied per detail level (finer → more aggressive)
_THRESHOLDS: tuple[float, float, float] = (0.6, 0.4, 0.2)

# Gain inside tanh for position sizing
_GAIN = 4.0

# Anomaly score above which we go flat
_ANOMALY_THRESH = 0.6

# Dead-band: weights with |w| below this collapse to zero
_DEAD_BAND = 0.04


def _soft_threshold(x: float, lam: float) -> float:
    """Shrink x toward 0 by lam; zero out if |x| <= lam."""
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


class _WaveletState:
    """Per-symbol EMA cascade state for three scales."""

    __slots__ = ("s0", "s1", "s2")

    def __init__(self) -> None:
        self.s0: float = 0.0  # smoothed at alpha=0.5
        self.s1: float = 0.0  # smoothed at alpha=0.25
        self.s2: float = 0.0  # smoothed at alpha=0.1


class WaveletDenoiseStrategy:
    """À trous wavelet denoising strategy with per-symbol EMA cascade state."""

    def __init__(self) -> None:
        self._states: dict[str, _WaveletState] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        if sym not in self._states:
            self._states[sym] = _WaveletState()
        ws = self._states[sym]

        mu = state.fwd_return_mean
        sigma = state.fwd_return_std

        # Update EMA cascade (causal: uses current observation, updates state)
        a0, a1, a2 = _ALPHAS
        ws.s0 = a0 * mu + (1.0 - a0) * ws.s0
        ws.s1 = a1 * mu + (1.0 - a1) * ws.s1
        ws.s2 = a2 * mu + (1.0 - a2) * ws.s2

        # Detail coefficients (difference between successive smooth levels)
        d0 = mu - ws.s0       # finest detail: raw minus fast-smooth
        d1 = ws.s0 - ws.s1    # mid-scale detail
        d2 = ws.s1 - ws.s2    # coarse detail

        # Adaptive threshold: scale * sigma (noise proportional to uncertainty)
        noise_ref = sigma + 1e-9
        t0, t1, t2 = _THRESHOLDS
        sd0 = _soft_threshold(d0, t0 * noise_ref)
        sd1 = _soft_threshold(d1, t1 * noise_ref)
        sd2 = _soft_threshold(d2, t2 * noise_ref)

        # Reconstruct denoised drift: low-frequency residual + denoised details
        denoised_drift = ws.s2 + sd2 + sd1 + sd0

        # Confidence
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Normalised signal for momentum field
        momentum = math.tanh(_GAIN * denoised_drift / (sigma + 1e-9)) * confidence

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=denoised_drift,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        raw = signals.momentum

        # Dead-band suppression
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh WaveletDenoiseStrategy instance."""
    return WaveletDenoiseStrategy()
