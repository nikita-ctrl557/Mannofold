"""Manifold-geometry strategy.

Reads only the :class:`ManifoldState` (position on the manifold, neighbourhood
forward-return statistics, anomaly, regime). Expected return comes from where the
state sits on the manifold; confidence is gated by how typical the state is.

To curb overtrading the target is shaped with several deterministic, per-symbol
mechanisms (all driven only by the stream of states seen so far):

* **hysteresis** — separate entry / exit thresholds on conviction, so a fresh
  position needs a stronger signal than is required to merely hold one;
* **target smoothing** — an EMA of the (gated) target weight across calls,
  damping bar-to-bar flip-flop;
* **a no-trade dead-band** — small targets collapse to zero;
* **gating** — entries require sufficient confidence and a non-anomalous,
  identified regime, while a sign-flip is treated as a fresh entry.

State is held per symbol on the instance and is therefore deterministic for a
given ordered stream of states. It depends only on past inputs, so it introduces
no lookahead. ``ManifoldStrategy()`` still constructs with zero arguments.
"""

from __future__ import annotations

import math

from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

_EPS = 1e-9


class ManifoldStrategy:
    def __init__(
        self,
        gain: float = 60.0,
        entry_threshold: float = 0.15,
        exit_threshold: float = 0.05,
        confidence_floor: float = 0.25,
        anomaly_gate: float = 0.6,
        smoothing: float = 0.35,
        deadband: float = 0.04,
    ):
        # `gain` maps the neighbourhood-Sharpe into a target weight via tanh.
        self._gain = gain
        # Hysteresis: a flat book must clear `entry_threshold`; an open one only
        # exits once conviction decays below `exit_threshold` (exit < entry).
        self._entry_threshold = entry_threshold
        self._exit_threshold = min(exit_threshold, entry_threshold)
        # Entries are gated on confidence and regime stability.
        self._confidence_floor = confidence_floor
        self._anomaly_gate = anomaly_gate
        # EMA smoothing factor in (0, 1]; lower = stickier target.
        self._smoothing = min(max(smoothing, _EPS), 1.0)
        # Targets whose magnitude is below the dead-band collapse to flat.
        self._deadband = deadband
        # Per-symbol smoothed target weight (deterministic instance state).
        self._ema: dict[str, float] = {}

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
        # Raw conviction from expected return, gated by confidence and de-grossed
        # off-manifold (preserved from the baseline).
        raw = math.tanh(self._gain * signals.expected_return)
        desired = raw * signals.confidence
        desired *= 1.0 - signals.anomaly

        prev = self._ema.get(signals.symbol, 0.0)

        # Hard gate: a flat or sign-flipping book must clear entry thresholds.
        # Anomalous / unidentified regimes or thin confidence force de-risking.
        gated = self._gate(desired, prev, signals)

        # Deterministic per-symbol EMA smoothing to damp churn.
        smoothed = prev + self._smoothing * (gated - prev)

        # Dead-band: collapse negligible exposure to flat to avoid micro-trades.
        if abs(smoothed) < self._deadband:
            smoothed = 0.0

        smoothed = max(-1.0, min(1.0, smoothed))
        self._ema[signals.symbol] = smoothed
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=smoothed)

    def _gate(self, desired: float, prev: float, signals: SignalSet) -> float:
        """Apply hysteresis + confidence/regime gating to the desired weight."""
        # Regime instability or anomaly ⇒ no fresh risk, decay toward flat.
        regime_ok = signals.regime_id != ANOMALY_REGIME
        stable = regime_ok and signals.anomaly <= self._anomaly_gate
        confident = signals.confidence >= self._confidence_floor

        magnitude = abs(desired)
        holding = abs(prev) > _EPS
        same_side = (desired >= 0.0) == (prev >= 0.0)

        if not stable:
            # Off-manifold / unknown regime: never add, only allow decay to flat.
            return 0.0

        if holding and same_side:
            # Already in a position on this side: hold unless conviction has
            # decayed below the (lower) exit threshold.
            if magnitude < self._exit_threshold:
                return 0.0
            return desired

        # Flat, or flipping sign (treated as a fresh entry): require both the
        # higher entry threshold and the confidence floor.
        if magnitude < self._entry_threshold or not confident:
            return 0.0
        return desired
