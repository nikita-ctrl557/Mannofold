"""Stochastic resonance strategy — nonlinear physics perspective.

In stochastic resonance (SR), a weak periodic signal becomes most detectable
not in the absence of noise, but at an OPTIMAL intermediate noise level σ*.
Too little noise: the signal never crosses detection threshold.
Too much noise:  the signal is buried in chaos.

The resonance gain R(σ) is a Gaussian bell curve in volatility (σ) space:

    R(σ) = exp(-((σ - σ*) / w)²)

peaked at σ* = 0.012 (moderate daily-return vol) with width w = 0.008.
This means the strategy sizes up when market volatility is at the "sweet spot"
where the drift signal is most reliably detectable, and shrinks toward zero in
dead-calm (σ → 0) or chaotic (σ → ∞) regimes.

Target weight formula:
    confidence = regime_prob * (1 - anomaly_score)
    sharpe     = fwd_return_mean / (fwd_return_std + eps)
    R          = exp(-((σ - σ*) / w)²)   [resonance gain]
    raw        = tanh(GAIN * tanh(sharpe)) * R * confidence
    weight     = 0 if |raw| < DEAD_BAND else clamp(raw, -1, 1)

Flat on ANOMALY_REGIME or anomaly_score > 0.6; dead-band |w| < 0.04 → 0.
No lookahead. build() takes no arguments.
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

NAME = "stochastic_resonance"
DESCRIPTION = (
    "Stochastic-resonance sizing: signal detectability peaks at an optimal "
    "intermediate volatility σ*=0.012. Resonance gain R(σ)=exp(-((σ-σ*)/w)²) "
    "upweights positions when vol is at the resonant level and suppresses both "
    "dead-calm and chaotic regimes. "
    "weight = tanh(gain*tanh(sharpe)) * R(fwd_return_std) * confidence, "
    "confidence = regime_prob*(1-anomaly_score). "
    "Flat on ANOMALY_REGIME or anomaly > 0.6; dead-band |w| < 0.04 → 0."
)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
_EPS = 1e-9

# Resonance bell-curve parameters (daily-return-std units)
_SIGMA_STAR = 0.012   # optimal noise level: moderate daily vol (~0.75% / day)
_WIDTH = 0.008        # half-width of resonance bell; tails off strongly outside

# Sizing parameters
_GAIN = 3.5           # outer tanh amplifier on the Sharpe-directional signal
_ANOMALY_CUTOFF = 0.6 # hard gate: anomaly_score above this → flat
_DEAD_BAND = 0.04     # collapse |weight| below this to zero (avoids churn)


def _resonance_gain(sigma: float) -> float:
    """Bell-curve resonance gain R(σ) = exp(-((σ - σ*) / w)²).

    Returns 1.0 at σ = σ* (optimal noise level) and decays toward 0 for
    very small (dead-calm) or very large (chaotic) volatility.
    """
    z = (sigma - _SIGMA_STAR) / _WIDTH
    return math.exp(-(z * z))


class StochasticResonanceStrategy:
    """Size positions proportional to the stochastic-resonance gain R(σ).

    signals():
        Encodes the Sharpe ratio as momentum and passes confidence + anomaly.

    target():
        Applies the resonance bell curve to the current volatility estimate,
        multiplies by the Sharpe-directional sizing and confidence.
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,                      # Sharpe ratio as directional signal
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates ─────────────────────────────────────────────────────────
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_CUTOFF:
            return flat

        # Recover volatility σ from stored Sharpe and expected_return ────────
        # sharpe = expected_return / (sigma + eps)  ⟹  sigma = |er / sharpe|
        sharpe = signals.momentum
        er = signals.expected_return
        if abs(sharpe) > _EPS:
            sigma = abs(er / sharpe)
        else:
            sigma = abs(er) + _EPS

        # Resonance gain: peaks at σ* = _SIGMA_STAR ──────────────────────────
        R = _resonance_gain(sigma)

        # Sharpe-directional sizing with resonance weighting ──────────────────
        raw = math.tanh(_GAIN * math.tanh(sharpe)) * R * signals.confidence

        # Dead-band suppression ───────────────────────────────────────────────
        if abs(raw) < _DEAD_BAND:
            return flat

        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh StochasticResonanceStrategy instance."""
    return StochasticResonanceStrategy()
