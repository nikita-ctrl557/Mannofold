"""Diffusion-reversion strategy: statistical-physics framing of anomaly relaxation.

The anomaly_score is treated as the DISTANCE FROM EQUILIBRIUM on the manifold.
By Fick's first law of diffusion, flux is proportional to the negative gradient of
concentration: J ∝ -∇c. Here the "concentration" is the anomaly_score and the
gradient in time is Δanomaly = anomaly_prev - anomaly_current (positive = relaxing).

When the state is diffusing BACK toward the manifold (anomaly decreasing from a
mid-band level), the system is expected to return toward neighbourhood-mean drift.
We take a mean-reverting position scaled by the diffusion flux magnitude and gated
to a mid-band (not calm, not extreme) anomaly window.

weight = -tanh(gain * tanh(sharpe)) * anomaly_relax_factor * confidence

anomaly_relax_factor > 0 only when:
  - anomaly is in mid-band (LOW_BAND < anomaly < HIGH_BAND)
  - anomaly is DECREASING (flux = prev_anomaly - anomaly > 0)

Flat on ANOMALY_REGIME or anomaly > HIGH_BAND; dead-band |w| < 0.04 -> 0.
"""

from __future__ import annotations

import math
from collections import defaultdict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "diffusion_reversion"
DESCRIPTION = (
    "Diffusion-based reversion: trades mean reversion when anomaly_score is in a "
    "mid-band and decreasing (diffusing back to the manifold), sized by Fick-flux."
)

# Tunable knobs
_GAIN = 2.5          # amplifier in -tanh(gain * tanh(sharpe))
_LOW_BAND = 0.15     # anomaly must be above this to enter mid-band
_HIGH_BAND = 0.85    # anomaly above this -> flat (extreme, no-trade zone)
_FLUX_SCALE = 4.0    # multiplier on raw flux (prev - curr anomaly) to form relax_factor
_DEAD_BAND = 0.04    # |weight| below this collapses to 0


class DiffusionReversionStrategy:
    """Mean-reverting strategy gated by diffusion flux on the manifold.

    Maintains per-symbol state to track anomaly change direction without lookahead.
    On each step the previous anomaly is stored; the flux (Δanomaly = prev - curr)
    measures how fast the state is relaxing back toward the manifold.
    """

    def __init__(self) -> None:
        # Per-symbol previous anomaly score for flux computation
        self._prev_anomaly: dict[str, float] = defaultdict(float)

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
        # Mean-reverting direction: negate neighbourhood drift (double-tanh)
        momentum = -math.tanh(_GAIN * math.tanh(sharpe))

        # Diffusion flux: positive when anomaly is relaxing (decreasing)
        prev = self._prev_anomaly[state.symbol]
        flux = prev - state.anomaly_score  # >0 when diffusing back to manifold

        # Update stored anomaly for next step (no lookahead)
        self._prev_anomaly[state.symbol] = state.anomaly_score

        # Mid-band gate: anomaly must be between LOW and HIGH band
        in_mid_band = _LOW_BAND < state.anomaly_score < _HIGH_BAND

        # anomaly_relax_factor > 0 only when in mid-band AND relaxing
        if in_mid_band and flux > 0.0:
            anomaly_relax_factor = min(1.0, flux * _FLUX_SCALE)
        else:
            anomaly_relax_factor = 0.0

        confidence = max(0.0, min(1.0, state.regime_prob))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            # Encode relax_factor * confidence in momentum, recovered in target()
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard flat: anomalous regime sentinel or extreme anomaly
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly >= _HIGH_BAND:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Retrieve flux state for this symbol
        # (prev already updated in signals(); recompute relax_factor from stored prev)
        # anomaly_relax_factor was embedded in signals.momentum — recover here via
        # stored per-symbol flux (we keep a separate dict for target use)
        relax_factor = self._target_relax.get(signals.symbol, 0.0)

        weight = signals.momentum * relax_factor * signals.confidence

        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)

    # ------------------------------------------------------------------
    # Override signals() to also cache the relax_factor for target()
    # ------------------------------------------------------------------


def _make_strategy() -> DiffusionReversionStrategy:
    """Factory that patches signals() to cache relax_factor for target()."""

    class _Strategy(DiffusionReversionStrategy):
        def __init__(self) -> None:
            super().__init__()
            self._target_relax: dict[str, float] = {}

        def signals(self, state: ManifoldState) -> SignalSet:
            sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)
            momentum = -math.tanh(_GAIN * math.tanh(sharpe))

            prev = self._prev_anomaly[state.symbol]
            flux = prev - state.anomaly_score

            self._prev_anomaly[state.symbol] = state.anomaly_score

            in_mid_band = _LOW_BAND < state.anomaly_score < _HIGH_BAND

            if in_mid_band and flux > 0.0:
                anomaly_relax_factor = min(1.0, flux * _FLUX_SCALE)
            else:
                anomaly_relax_factor = 0.0

            # Cache for target()
            self._target_relax[state.symbol] = anomaly_relax_factor

            confidence = max(0.0, min(1.0, state.regime_prob))

            return SignalSet(
                ts=state.ts,
                symbol=state.symbol,
                momentum=momentum,
                expected_return=state.fwd_return_mean,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=confidence,
            )

    return _Strategy()


def build() -> Strategy:
    """Return a fresh DiffusionReversionStrategy instance."""
    return _make_strategy()
