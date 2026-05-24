"""Meta-engine: performance-weighted blend of sub-engines.

Instead of picking one sub-engine (bandit), this softmax-weights all of them by
their recent (past-only) directional reward and blends their target weights — a
smoother regime-adaptive ensemble. Blind by construction.
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import ANOMALY_REGIME, ManifoldState, SignalSet, TargetPosition
from mannofold.signals.strategies.momentum_velocity import build as _mom
from mannofold.signals.strategies.density_gated import build as _dens
from mannofold.signals.strategies.mean_reversion import build as _rev
from mannofold.signals.strategies.kelly_capped import build as _kelly
from mannofold.signals.strategies.manifold_core import build as _core

NAME = "engine_blend_meta"
DESCRIPTION = "Softmax performance-weighted blend of sub-engines (regime-adaptive ensemble, no-lookahead)."

_ALPHA = 0.05
_TEMP = 0.25


class EngineBlendMeta:
    def __init__(self) -> None:
        self._subs = {
            "mom": _mom(), "dens": _dens(), "rev": _rev(),
            "kelly": _kelly(), "core": _core(),
        }
        self._reward: dict[str, dict[str, float]] = {}
        self._prev_w: dict[str, dict[str, float]] = {}
        self._prev_drift: dict[str, float] = {}
        self._pending: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        targets = {n: s.target(s.signals(state)).target_weight for n, s in self._subs.items()}
        rew = self._reward.setdefault(sym, {n: 0.0 for n in self._subs})
        prev_w = self._prev_w.get(sym)
        if prev_w is not None and sym in self._prev_drift:
            move = state.fwd_return_mean - self._prev_drift[sym]
            for n, pw in prev_w.items():
                r = 1.0 if (pw > 0 and move > 0) or (pw < 0 and move < 0) else (-1.0 if pw != 0 else 0.0)
                rew[n] = _ALPHA * r + (1 - _ALPHA) * rew[n]
        self._prev_w[sym] = targets
        self._prev_drift[sym] = state.fwd_return_mean

        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > 0.6:
            choice = 0.0
        else:
            mx = max(rew.values())
            exps = {n: math.exp((rew[n] - mx) / _TEMP) for n in self._subs}
            z = sum(exps.values()) or 1.0
            choice = sum(exps[n] / z * targets[n] for n in self._subs)
        self._pending[sym] = max(-1.0, min(1.0, choice))
        return SignalSet(
            ts=state.ts, symbol=sym, momentum=targets["mom"],
            expected_return=state.fwd_return_mean, anomaly=state.anomaly_score,
            regime_id=state.regime_id, confidence=state.regime_prob * (1.0 - state.anomaly_score),
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        return TargetPosition(ts=signals.ts, symbol=signals.symbol,
                              target_weight=self._pending.get(signals.symbol, 0.0))


def build() -> Strategy:
    return EngineBlendMeta()
