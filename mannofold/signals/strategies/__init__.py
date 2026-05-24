"""Pluggable strategy-variant registry.

Each sibling module in this package defines exactly one strategy variant and
exposes three module-level symbols:

    NAME: str          # short unique id, e.g. "momentum_velocity"
    DESCRIPTION: str   # one line describing the edge
    def build() -> Strategy: ...   # returns a fresh strategy instance

A strategy implements the ``mannofold.contracts.interfaces.Strategy`` Protocol:
``signals(state: ManifoldState) -> SignalSet`` and
``target(signals: SignalSet) -> TargetPosition``.

``discover()`` imports every sibling module and returns the registry so the
optimizer can backtest all variants without hard-coding their names.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

from mannofold.contracts.interfaces import Strategy


@dataclass(frozen=True)
class StrategyEntry:
    name: str
    description: str
    build: Callable[[], Strategy]


def discover() -> list[StrategyEntry]:
    """Import every sibling module and collect its NAME/DESCRIPTION/build()."""
    entries: list[StrategyEntry] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except Exception:
            # Skip a module that fails to import (e.g. mid-write) rather than
            # breaking discovery for every caller.
            continue
        name = getattr(mod, "NAME", None)
        build = getattr(mod, "build", None)
        if not name or not callable(build):
            continue
        entries.append(
            StrategyEntry(
                name=name,
                description=getattr(mod, "DESCRIPTION", ""),
                build=build,
            )
        )
    return sorted(entries, key=lambda e: e.name)
