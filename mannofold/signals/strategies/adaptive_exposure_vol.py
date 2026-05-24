"""Adaptive Exposure Vol strategy.

Month-over-month consistency via classic volatility control: keep a per-symbol
EMA of realised signal volatility (proxied by fwd_return_std) and set total
exposure = clamp(target_vol / (ema_vol + eps), 0, 1) so the book runs at a
roughly constant risk level regardless of the prevailing vol regime.

Signal:   sharpe = fwd_return_mean / (fwd_return_std + eps)
Direction: sign(sharpe)
Sizing:   weight = sign(sharpe) * tanh(gain * |tanh(sharpe)|) * exposure * confidence
Exposure: clamp(target_vol / (ema_vol + eps), 0, 1)
Confidence: regime_prob * (1 - anomaly_score)
Gates:    flat on ANOMALY_REGIME or anomaly_score > 0.6
          dead-band |w| < 0.04 -> 0
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

NAME = "adaptive_exposure_vol"
DESCRIPTION = (
    "Classic volatility-control strategy for month-over-month consistency: "
    "per-symbol EMA of realised vol (fwd_return_std proxy) scales total exposure "
    "to target_vol, keeping book risk roughly constant; direction from Sharpe ratio."
)

_EPS = 1e-9
_GAIN = 2.5
_ANOMALY_GATE = 0.6
_DEADBAND = 0.04
_EMA_ALPHA = 0.15
_TARGET_VOL = 0.02


class AdaptiveExposureVolStrategy:
    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        ema_alpha: float = _EMA_ALPHA,
        target_vol: float = _TARGET_VOL,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._ema_alpha = ema_alpha
        self._target_vol = target_vol
        # Per-symbol EMA of fwd_return_std (initialised lazily on first tick).
        self._ema_vol: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # --- Update per-symbol EMA of fwd_return_std (no lookahead) ---
        prev_ema = self._ema_vol.get(sym, state.fwd_return_std)
        new_ema = self._ema_alpha * state.fwd_return_std + (1.0 - self._ema_alpha) * prev_ema
        self._ema_vol[sym] = new_ema

        # --- Sharpe-based direction ---
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # --- Vol-control exposure scalar ---
        exposure = min(1.0, max(0.0, self._target_vol / (new_ema + _EPS)))

        # --- Confidence: regime stability attenuated by anomaly ---
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        # Store sharpe in momentum; exposure folded into confidence via momentum field.
        # We store exposure*confidence product in confidence and raw sharpe in momentum
        # so target() can reconstruct the full weight.
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * exposure,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        sharpe = signals.momentum
        # weight = sign(sharpe) * tanh(gain * |tanh(sharpe)|) * exposure * confidence
        direction = math.copysign(1.0, sharpe) if sharpe != 0.0 else 0.0
        magnitude = math.tanh(self._gain * abs(math.tanh(sharpe)))
        weight = direction * magnitude * signals.confidence

        # Dead-band: avoid noise trading near zero.
        if abs(weight) < self._deadband:
            weight = 0.0

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return AdaptiveExposureVolStrategy()
