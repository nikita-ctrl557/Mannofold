"""Frozen contracts: the single source of truth shared across every workstream.

Nothing in this package may import from `mannofold.feed`, `mannofold.features`,
`mannofold.manifold`, `mannofold.signals`, `mannofold.engine`, `mannofold.persist`
or `mannofold.api`. Dependencies flow *toward* contracts, never out of it.
"""

from mannofold.contracts.events import EventType, StreamEvent
from mannofold.contracts.interfaces import (
    DataFeed,
    FeaturePipeline,
    ManifoldModel,
    RiskSizer,
    StateStore,
    Strategy,
)
from mannofold.contracts.models import (
    Bar,
    FeatureVector,
    Fill,
    ManifoldState,
    Mode,
    Order,
    PortfolioState,
    Regime,
    Side,
    SignalSet,
    StepResult,
    TargetPosition,
)

__all__ = [
    "Bar",
    "FeatureVector",
    "Fill",
    "ManifoldState",
    "Mode",
    "Order",
    "PortfolioState",
    "Regime",
    "Side",
    "SignalSet",
    "StepResult",
    "TargetPosition",
    "DataFeed",
    "FeaturePipeline",
    "ManifoldModel",
    "RiskSizer",
    "StateStore",
    "Strategy",
    "EventType",
    "StreamEvent",
]
