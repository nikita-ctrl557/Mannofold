"""Fluctuation-Dissipation strategy.

Statistical-mechanics analogy: the fluctuation-dissipation theorem states that a
system's linear-response susceptibility χ is proportional to the variance of its
equilibrium fluctuations divided by the temperature.

Here we map:
  - "fluctuations" -> EMA variance of the drift (fwd_return_mean) over time
  - "temperature"  -> current dispersion fwd_return_std  (volatility proxy)
  - susceptibility χ = ema_var_drift / (fwd_return_std + eps)

High χ: the state's drift varies a lot relative to its local noise — the manifold
responds strongly and predictably to perturbations, which is exactly when a
drift-following bet has the best signal-to-noise ratio.

target_weight = tanh(gain * tanh(sharpe)) * clamp(χ_normalized, 0, 1) * confidence
confidence    = regime_prob * (1 - anomaly_score)
Flat on ANOMALY_REGIME or anomaly_score > 0.6; dead-band |w| < 0.04 -> 0.
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

NAME = "fluctuation_dissipation"
DESCRIPTION = (
    "Fluctuation-dissipation susceptibility filter: trades drift only when the "
    "manifold's response function χ = EMA_var(drift) / dispersion is high, "
    "indicating a predictably responsive regime."
)

_EPS = 1e-9
_GAIN = 3.0           # gain inside outer tanh(gain * tanh(sharpe))
_ANOMALY_GATE = 0.6   # anomaly_score above this -> flat immediately
_DEADBAND = 0.04      # dead-band: |weight| below this -> 0
# EMA smoothing factor for the drift variance estimator (α for EMA of squares).
# α = 2/(N+1) with N≈19 gives a ~20-bar half-life.
_EMA_ALPHA = 0.095
# Normalisation percentile proxy: we use a running EMA of χ itself to normalise.
_CHI_EMA_ALPHA = 0.05   # slower EMA for normalising χ (~39-bar half-life)


class FluctuationDissipationStrategy:
    """Per-symbol EMA state; no lookahead; pure online updates."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
        chi_ema_alpha: float = _CHI_EMA_ALPHA,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._alpha = ema_alpha
        self._chi_alpha = chi_ema_alpha

        # per-symbol state
        self._ema_drift: dict[str, float] = {}        # EMA of fwd_return_mean
        self._ema_var_drift: dict[str, float] = {}    # EMA of squared deviation
        self._ema_chi: dict[str, float] = {}          # EMA of raw χ for normalisation

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        drift = state.fwd_return_mean

        # -- Update EMA variance of drift (fluctuation estimator) --------
        prev_ema = self._ema_drift.get(sym, drift)
        new_ema = (1.0 - self._alpha) * prev_ema + self._alpha * drift
        self._ema_drift[sym] = new_ema

        deviation = drift - prev_ema          # deviation from previous mean
        prev_var = self._ema_var_drift.get(sym, deviation ** 2)
        new_var = (1.0 - self._alpha) * prev_var + self._alpha * (deviation ** 2)
        self._ema_var_drift[sym] = new_var

        # -- Susceptibility χ = variance_of_drift / temperature ----------
        temperature = state.fwd_return_std + _EPS
        chi_raw = new_var / temperature

        # -- Normalise χ with a slow EMA so it stays in [0, 1] ----------
        prev_chi_ema = self._ema_chi.get(sym, chi_raw)
        chi_ema = (1.0 - self._chi_alpha) * prev_chi_ema + self._chi_alpha * chi_raw
        self._ema_chi[sym] = chi_ema

        # χ_normalized: ratio to running mean; clamp to [0, 1]
        chi_normalized = chi_raw / (chi_ema + _EPS)
        chi_normalized = max(0.0, min(1.0, chi_normalized))

        # -- Sharpe-normalised drift as the directional signal -----------
        sharpe = drift / (state.fwd_return_std + _EPS)

        # -- Confidence --------------------------------------------------
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Pack into SignalSet; momentum carries χ_normalized * tanh(tanh(sharpe))
        # so that target() can do a single multiply.
        inner = math.tanh(sharpe)
        outer = math.tanh(self._gain * inner)
        composite_momentum = outer * chi_normalized   # in (-1, 1)

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=composite_momentum,
            expected_return=drift,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = signals.momentum * signals.confidence

        if abs(weight) < self._deadband:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh FluctuationDissipationStrategy instance."""
    return FluctuationDissipationStrategy()
