"""Crisis Long-Vol strategy: amplify exposure exactly when anomaly/vol is highest.

Unlike most strategies that go flat during crashes, this strategy treats elevated
anomaly scores and high forward-return vol as the primary engagement signal —
capturing large drift moves during regime shifts and high-volatility crashes.
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "crisis_long_vol"
DESCRIPTION = "Long-volatility crisis alpha: amplifies exposure during anomaly/regime-shift episodes to capture crash drift."

# Tunable knobs
_GAIN = 2.0        # double-tanh amplifier: tanh(gain * tanh(sharpe))
_K = 1.8           # anomaly amplification factor: multiply by (1 + k * anomaly_score)
_VOL_K = 1.2       # vol amplification: multiply by (1 + vol_k * norm_vol)
_VOL_REF = 0.02    # reference vol level for normalisation (annualised-ish daily)
_DEAD_BAND = 0.04  # suppress |weight| below this to zero
_MIN_ANOMALY = 0.1 # below this anomaly score, scale back to avoid noise regime


class CrisisLongVolStrategy:
    """Long-vol crisis alpha: size UP when anomaly_score and fwd_return_std are elevated.

    Mechanics:
    - Base signal: double-tanh of neighbourhood Sharpe, bounding output to (-1, 1).
    - Anomaly amplifier: multiply by (1 + K * anomaly_score) — more off-manifold
      means larger position, not smaller, to capture fat-tail drift.
    - Vol amplifier: multiply by (1 + VOL_K * norm_vol) — higher vol regime means
      bigger crash moves; we want more exposure to the drift direction.
    - Calm-market damper: when anomaly_score < MIN_ANOMALY, scale back linearly so
      we stay small/flat in boring regimes and avoid noise trading.
    - Optional regime_prob gate: used lightly (square-root) to avoid over-suppressing
      in anomaly episodes where regime_prob is naturally low.
    - Final clamp to [-1, 1] and dead-band |w| < DEAD_BAND -> 0.
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        # Neighbourhood Sharpe: expected forward return per unit of its spread
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Double-tanh base signal bounds result to (-1, 1)
        base = math.tanh(_GAIN * math.tanh(sharpe))

        # Anomaly amplification: INCREASE size as state drifts off-manifold
        anomaly_amp = 1.0 + _K * state.anomaly_score

        # Vol amplification: normalise fwd_return_std by reference level
        norm_vol = state.fwd_return_std / _VOL_REF
        vol_amp = 1.0 + _VOL_K * min(norm_vol, 3.0)  # cap to avoid runaway

        # Calm-market damper: ramp from 0 to 1 as anomaly_score goes from 0 to MIN_ANOMALY
        # This keeps us flat/small in low-anomaly, low-vol regimes
        calm_damper = min(1.0, state.anomaly_score / _MIN_ANOMALY)

        # Light confidence gate: sqrt to avoid over-suppressing in anomaly episodes
        # where regime_prob is naturally depressed
        confidence = math.sqrt(max(0.0, min(1.0, state.regime_prob))) if state.regime_prob > 0 else 0.5

        momentum = base * anomaly_amp * vol_amp * calm_damper

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Apply light confidence gate
        weight = signals.momentum * signals.confidence

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh CrisisLongVolStrategy instance."""
    return CrisisLongVolStrategy()
