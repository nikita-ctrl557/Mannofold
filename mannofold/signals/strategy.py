"""Manifold-geometry strategy.

Reads only the :class:`ManifoldState` (position on the manifold, neighbourhood
forward-return statistics, anomaly, regime). Expected return comes from where the
state sits on the manifold; confidence is gated by how typical the state is.
"""

from __future__ import annotations

import math

from mannofold.contracts.models import ManifoldState, SignalSet, TargetPosition

_EPS = 1e-9


class ManifoldStrategy:
    def __init__(self, gain: float = 60.0):
        # `gain` maps the neighbourhood-Sharpe into a target weight via tanh.
        self._gain = gain

    def signals(self, state: ManifoldState) -> SignalSet:
        # Neighbourhood "Sharpe": expected forward return per unit of its spread.
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
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
        raw = math.tanh(self._gain * signals.expected_return)
        weight = raw * signals.confidence
        # De-gross hard when off-manifold.
        weight *= 1.0 - signals.anomaly
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)
