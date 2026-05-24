"""Meta-engine: online multi-armed bandit over sub-engines.

Holds several proven sub-engines and, each bar, picks the one with the best
recent reward (an EMA of how well that engine's PREVIOUS-bar direction matched
the subsequently-observed drift change). Strictly causal / no-lookahead: rewards
use only past information. All sub-engines are stepped every bar.
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

NAME = "engine_bandit_meta"
DESCRIPTION = "Online bandit: each bar picks the sub-engine with the best recent (past-only) directional reward."

_ALPHA = 0.05  # reward EMA rate


class EngineBanditMeta:
    def __init__(self) -> None:
        self._subs = {
            "mom": _mom(), "dens": _dens(), "rev": _rev(),
            "kelly": _kelly(), "core": _core(),
        }
        self._reward: dict[str, dict[str, float]] = {}     # sym -> {name: ema reward}
        self._prev_w: dict[str, dict[str, float]] = {}      # sym -> {name: prev target}
        self._prev_drift: dict[str, float] = {}
        self._pending: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        targets = {n: s.target(s.signals(state)).target_weight for n, s in self._subs.items()}

        # Update rewards using only PAST info: did each sub's previous-bar
        # direction match the now-observed change in neighbourhood drift?
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
            best = max(rew, key=lambda n: rew[n])
            choice = targets[best]
        self._pending[sym] = choice
        return SignalSet(
            ts=state.ts, symbol=sym, momentum=targets["mom"],
            expected_return=state.fwd_return_mean, anomaly=state.anomaly_score,
            regime_id=state.regime_id, confidence=state.regime_prob * (1.0 - state.anomaly_score),
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        return TargetPosition(ts=signals.ts, symbol=signals.symbol,
                              target_weight=self._pending.get(signals.symbol, 0.0))


def build() -> Strategy:
    return EngineBanditMeta()
