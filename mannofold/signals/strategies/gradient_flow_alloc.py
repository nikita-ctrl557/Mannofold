"""Gradient-flow allocation strategy: natural-gradient flow toward mean-variance optimum.

Treats the position w as evolving by gradient ascent on a mean-variance utility:
    U(w) = w·μ − 0.5·η·w²·σ²
    dU/dw = μ − η·w·σ²
    w* = μ/(η·σ²)  (Kelly-like fixed point)

Instead of jumping to w*, we FLOW toward it each step:
    w_new = w_prev + lr·(μ − η·w_prev·σ²)
    target_weight = tanh(gain·w_new) · confidence

This provides smooth, stable allocation that converges to the mean-variance optimum
with a controlled step size, reducing turnover and whipsaw relative to direct Kelly.
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

NAME = "gradient_flow_alloc"
DESCRIPTION = (
    "Natural-gradient flow of target weight along the expected-utility gradient "
    "with quadratic risk penalty (mean-variance); flows toward Kelly fixed-point "
    "w*=μ/(η·σ²) with step lr, squashed tanh(gain·w_flow)·confidence."
)

_EPS = 1e-9
# Risk-aversion coefficient (η): scales the variance penalty in U(w).
_ETA = 2.0
# Gradient-flow learning rate: fraction of the gradient applied each step.
_LR = 0.20
# tanh squash gain applied to the flowing weight.
_GAIN = 2.5
# Anomaly gate: go flat if anomaly_score exceeds this threshold.
_ANOMALY_GATE = 0.6
# Dead-band: zero out tiny positions to avoid noise trading.
_DEADBAND = 0.04


class GradientFlowAllocStrategy:
    def __init__(
        self,
        eta: float = _ETA,
        lr: float = _LR,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ):
        self._eta = eta
        self._lr = lr
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        # Per-symbol gradient-flow state: w_prev for each symbol.
        self._w_flow: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        mu = state.fwd_return_mean
        sigma2 = state.fwd_return_std ** 2 + _EPS

        # Gradient of U(w) w.r.t. w: dU/dw = μ − η·w·σ²
        sym = state.symbol
        w_prev = self._w_flow.get(sym, 0.0)
        grad = mu - self._eta * w_prev * sigma2

        # Flow step: w_new = w_prev + lr · grad
        w_new = w_prev + self._lr * grad

        # Squash to (-1, 1) with gain — this is the flowing momentum signal.
        momentum = math.tanh(self._gain * w_new)

        # Confidence: regime stability attenuated by anomaly proximity.
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat, decay state.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            # Decay w_flow toward zero to avoid stale state on re-entry.
            self._w_flow[sym] = 0.9 * self._w_flow.get(sym, 0.0)
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Recompute w_new from stored w_prev so target stays consistent with signals.
        # (signals() already computed the flow step; we commit it here.)
        # Retrieve the pre-step w_prev by inverting: w_new stored as tanh output is
        # not invertible, so we track w_flow directly alongside.
        # Update stored flow state using the squashed value as a proxy anchor.
        # w_flow tracks the un-squashed flowing weight.
        w_prev = self._w_flow.get(sym, 0.0)
        # Recompute the same gradient and step that signals() used.
        mu = signals.expected_return
        # We don't have sigma2 here directly, but we can back-compute from momentum:
        # momentum = tanh(gain * w_new), so w_new = atanh(momentum) / gain
        w_new = math.atanh(max(-0.9999, min(0.9999, signals.momentum))) / self._gain
        self._w_flow[sym] = w_new

        # Target weight: squashed flow scaled by confidence.
        raw_weight = signals.momentum * signals.confidence

        # Dead-band: avoid noise trades near zero.
        weight = 0.0 if abs(raw_weight) < self._deadband else raw_weight

        # Hard clip to [-1, 1].
        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return GradientFlowAllocStrategy()
