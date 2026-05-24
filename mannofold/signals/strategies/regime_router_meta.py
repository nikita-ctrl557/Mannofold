"""Meta-engine: route to the best sub-engine for the current regime.

A "fund of engines" — it holds instances of several proven sub-engines and, at
each bar, delegates to whichever one suits the CURRENT regime/volatility/anomaly
(static, hand-mapped routing). It is blind by construction: the routing decision
uses only the current ManifoldState, and every sub-engine is itself no-lookahead.
All sub-engines are stepped every bar so their internal state stays consistent.
"""

from __future__ import annotations

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import ANOMALY_REGIME, ManifoldState, SignalSet, TargetPosition
from mannofold.signals.strategies.momentum_velocity import build as _mom
from mannofold.signals.strategies.density_gated import build as _dens
from mannofold.signals.strategies.mean_reversion import build as _rev
from mannofold.signals.strategies.manifold_core import build as _core

NAME = "regime_router_meta"
DESCRIPTION = "Routes to momentum (high-vol/crash), density-gated (calm), mean-rev (choppy), core (default) by regime."


class RegimeRouterMeta:
    def __init__(self) -> None:
        self._mom = _mom()
        self._dens = _dens()
        self._rev = _rev()
        self._core = _core()
        self._vol_ema: dict[str, float] = {}
        self._pending: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        # Step every sub-engine each bar so per-symbol state stays warm.
        targets = {
            "mom": self._mom.target(self._mom.signals(state)).target_weight,
            "dens": self._dens.target(self._dens.signals(state)).target_weight,
            "rev": self._rev.target(self._rev.signals(state)).target_weight,
            "core": self._core.target(self._core.signals(state)).target_weight,
        }
        sym = state.symbol
        v = self._vol_ema.get(sym)
        v = state.fwd_return_std if v is None else 0.05 * state.fwd_return_std + 0.95 * v
        self._vol_ema[sym] = v
        hot = state.fwd_return_std > 1.2 * (v + 1e-9)

        # Route on current state only (causal):
        if state.regime_id == ANOMALY_REGIME:
            choice = 0.0  # off-manifold -> flat
        elif hot or state.anomaly_score > 0.45:
            choice = targets["mom"]      # high-vol / stress -> momentum
        elif state.density >= 1.0:
            choice = targets["dens"]     # typical / calm -> density-gated
        elif state.fwd_return_std > (v + 1e-9):
            choice = targets["rev"]      # elevated-but-not-stress dispersion -> mean-revert
        else:
            choice = targets["core"]     # default
        self._pending[sym] = choice
        return SignalSet(
            ts=state.ts, symbol=sym,
            momentum=targets["mom"], expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score, regime_id=state.regime_id,
            confidence=state.regime_prob * (1.0 - state.anomaly_score),
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        w = self._pending.get(signals.symbol, 0.0)
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    return RegimeRouterMeta()
