"""Baseline: the shipped manifold-geometry strategy, registered as a variant.

Serves as the reference template for new variants: expose NAME, DESCRIPTION and
build(); implement signals()/target() reading manifold geometry.
"""

from __future__ import annotations

from mannofold.contracts.interfaces import Strategy
from mannofold.signals.strategy import ManifoldStrategy

NAME = "manifold_core"
DESCRIPTION = "Baseline: neighbourhood-Sharpe conviction with hysteresis + vol smoothing."


def build() -> Strategy:
    return ManifoldStrategy()
