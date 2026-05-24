"""COVID Crash Specialist: momentum engine that engages hard in high-volatility regimes.

Tuned for the COVID/2018-vol crash scenarios where momentum_velocity achieved Sharpe +2.0.
Maintains a per-symbol EMA of fwd_return_std to distinguish crash (high-vol) vs calm regimes.
A logistic gate is ~1 in high-vol/crash conditions and ~0.2 in calm markets.
Does NOT flat on moderate anomaly (crashes are anomalous by definition) — only flats on
ANOMALY_REGIME. Caps size during high anomaly instead.
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

NAME = "covid_crash_specialist"
DESCRIPTION = (
    "Crash-regime momentum specialist: logistic vol-gate engages hard when current "
    "fwd_return_std exceeds its EMA (high-vol/crash), stays small in calm markets. "
    "Flat only on ANOMALY_REGIME; caps but does not flatten on moderate anomaly."
)

# Tunable knobs
_GAIN = 3.0           # amplifier inside outer tanh — more aggressive than momentum_velocity
_EMA_ALPHA = 0.05     # per-symbol EMA decay for fwd_return_std baseline
_GATE_K = 12.0        # logistic steepness: how sharply the gate opens vs EMA
_GATE_MIN = 0.2       # floor gate value in calm conditions
_GATE_MAX = 1.0       # ceiling gate value in crash conditions
_ANOMALY_CAP = 0.5    # when anomaly_score > this, cap |weight| to this fraction
_DEAD_BAND = 0.04     # collapse |weight| below this to 0


class CovidCrashSpecialist:
    """Crash-regime momentum strategy with adaptive volatility gate.

    Per-symbol EMA of fwd_return_std serves as the calm baseline.
    A logistic function of (current_std / ema_std - 1) produces a gate in [0.2, 1.0]:
      - gate ~ 1.0  when current vol is elevated vs its EMA  (crash/high-vol regime)
      - gate ~ 0.2  when current vol is at or below its EMA  (calm regime)

    weight = tanh(gain * tanh(sharpe)) * gate * confidence
    where confidence = regime_prob (not penalised by anomaly_score).
    """

    def __init__(self) -> None:
        # Per-symbol EMA state: symbol -> ema of fwd_return_std
        self._vol_ema: dict[str, float] = defaultdict(lambda: 0.0)
        self._initialised: dict[str, bool] = defaultdict(lambda: False)

    def _update_vol_ema(self, symbol: str, current_std: float) -> float:
        """Update and return the EMA of fwd_return_std for the given symbol."""
        if not self._initialised[symbol]:
            self._vol_ema[symbol] = current_std
            self._initialised[symbol] = True
        else:
            self._vol_ema[symbol] = (
                _EMA_ALPHA * current_std + (1.0 - _EMA_ALPHA) * self._vol_ema[symbol]
            )
        return self._vol_ema[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        # Neighbourhood Sharpe
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Update vol EMA and compute logistic gate
        ema_std = self._update_vol_ema(state.symbol, state.fwd_return_std)
        ema_ref = max(ema_std, 1e-9)
        # ratio > 1 means current vol exceeds baseline → crash/high-vol regime
        ratio = state.fwd_return_std / ema_ref - 1.0
        logistic = 1.0 / (1.0 + math.exp(-_GATE_K * ratio))
        gate = _GATE_MIN + (_GATE_MAX - _GATE_MIN) * logistic

        # Confidence = regime_prob only (not penalised by anomaly_score — crashes ARE anomalous)
        confidence = max(0.0, min(1.0, state.regime_prob))

        # Core momentum signal: double-tanh of Sharpe
        momentum = math.tanh(_GAIN * math.tanh(sharpe)) * gate * confidence

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
        # Flat only on the ANOMALY_REGIME sentinel (manifold is undefined)
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        weight = signals.momentum

        # Cap (not flatten) on high anomaly_score — crashes are anomalous but tradeable
        if signals.anomaly > _ANOMALY_CAP:
            cap = 1.0 - (signals.anomaly - _ANOMALY_CAP) / (1.0 - _ANOMALY_CAP + 1e-9)
            cap = max(0.1, cap)  # never cap below 0.1
            weight = max(-cap, min(cap, weight))

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh CovidCrashSpecialist instance."""
    return CovidCrashSpecialist()
