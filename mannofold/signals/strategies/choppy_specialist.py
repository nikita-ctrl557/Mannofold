"""Choppy-market specialist strategy.

Fades drift (mean-reversion) ONLY when the regime is choppy / range-bound,
identified by a low signal-to-noise ratio (std >> |mean|). Goes flat when the
market is in a clean trend, anomalous, or when the anomaly score is high.

Best suited for use-case specialisation in high-dispersion, range-bound regimes
where mean_reversion historically wins (e.g. AAPL choppy scenarios).
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

NAME = "choppy_specialist"
DESCRIPTION = (
    "Fade the drift only in choppy/range-bound regimes (low signal-to-noise); "
    "flat during clean trends or anomalous states."
)

_EPS = 1e-9
_GAIN = 3.0           # outer tanh gain on the Sharpe-based fade signal
_ANOMALY_GATE = 0.6   # anomaly_score above this -> flat
_DEADBAND = 0.04      # |weight| below this collapses to 0

# Per-symbol EMA decay for running fwd_return_std
_STD_ALPHA = 0.05     # slow EMA (~20-bar half-life)

# Choppiness gate logistic parameters.
# choppy_gate = sigmoid(scale * (snr_inv - mid))
# snr_inv = std / (|mean| + eps)  ... high when drift is small relative to noise
# When snr_inv > mid -> range-bound/choppy -> gate ~ 1 -> revert
# When snr_inv < mid -> clean trend -> gate ~ 0 -> flat
_CHOPPY_MID = 1.5     # pivot: std / |mean| ~ 1.5 is the "choppy" threshold
_CHOPPY_SCALE = 2.0   # logistic steepness


def _logistic(x: float, scale: float, mid: float) -> float:
    """Smooth sigmoid gate, output in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-scale * (x - mid)))


class ChoppySpecialistStrategy:
    """Mean-reversion specialist for choppy / high-dispersion regimes.

    weight = -tanh(gain * tanh(sharpe)) * choppy_gate * confidence

    where:
      sharpe     = fwd_return_mean / (fwd_return_std + eps)
      snr_inv    = fwd_return_std / (|fwd_return_mean| + eps)   (inverse SNR)
      choppy_gate = sigmoid(choppy_scale * (snr_inv - choppy_mid))
      confidence = regime_prob * (1 - anomaly_score)

    Goes flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - choppy_gate is near 0 (clean trend)
    """

    def __init__(self) -> None:
        # Per-symbol EMA of fwd_return_std to normalise the choppiness measure
        self._std_ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        std = state.fwd_return_std
        mean = state.fwd_return_mean

        # Update per-symbol EMA of dispersion (std)
        if sym not in self._std_ema:
            self._std_ema[sym] = max(std, _EPS)
        else:
            self._std_ema[sym] = (
                _STD_ALPHA * std + (1.0 - _STD_ALPHA) * self._std_ema[sym]
            )
        ema_std = max(self._std_ema[sym], _EPS)

        # Inverse SNR: high when drift is small relative to dispersion -> choppy
        # We normalise std by its EMA so the ratio is relative (avoids level bias)
        rel_std = std / ema_std          # > 1 means currently more volatile than usual
        abs_mean_norm = abs(mean) / (ema_std + _EPS)  # |drift| normalised by typical noise
        snr_inv = rel_std / (abs_mean_norm + _EPS)

        # Choppiness gate: 1 in choppy regimes, 0 in clean trends
        choppy_gate = _logistic(snr_inv, _CHOPPY_SCALE, _CHOPPY_MID)

        # Sharpe and momentum from neighbourhood perspective
        sharpe = mean / (std + _EPS)
        momentum = math.tanh(sharpe)  # in (-1, 1)

        # Confidence: regime certainty * cleanliness (no anomaly)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Encode choppiness-gated momentum for target(): multiply into momentum
        gated_momentum = momentum * choppy_gate

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=gated_momentum,
            expected_return=mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian fade: negate the (already choppiness-gated) momentum
        # weight = -tanh(gain * gated_momentum) * confidence
        raw_fade = -math.tanh(_GAIN * signals.momentum)
        weight = raw_fade * signals.confidence

        # Dead-band: suppress micro positions
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ChoppySpecialistStrategy instance."""
    return ChoppySpecialistStrategy()
