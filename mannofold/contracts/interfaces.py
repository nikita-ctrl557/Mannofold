"""Protocol interfaces — the seams the parallel workstreams build against.

Every implementation lives in its own subpackage and depends ONLY on these
Protocols + the models. The engine wires concrete implementations together but
never imports their modules directly beyond construction.

Numeric boundary convention
----------------------------
``FeaturePipeline`` emits domain ``FeatureVector`` objects. The engine flattens
``.values`` into a ``numpy`` matrix for ``ManifoldModel`` (which is purely
numeric and ts/symbol-agnostic), then re-attaches ``ts``/``symbol`` to the
returned :class:`ManifoldState`. This keeps the manifold math decoupled from the
domain types.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mannofold.contracts.models import (
    Bar,
    FeatureVector,
    ManifoldState,
    Mode,
    Order,
    PortfolioState,
    Regime,
    SignalSet,
    StepResult,
    TargetPosition,
)

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class DataFeed(Protocol):
    """Yields bars in time order. The ONLY thing that drives the clock forward.

    Backtest and paper modes differ only in this object: ``HistoricalReplayFeed``
    yields as fast as possible; ``LiveReplayFeed``/``AlpacaFeed`` yield on a
    wall-clock cadence. Neither may ever yield a future bar early.
    """

    mode: Mode

    def stream(self) -> Iterator[Bar]: ...


class FeaturePipeline(Protocol):
    """Builds x_t from a trailing bar window. Owns its scaler.

    ``fit`` must be called on TRAIN bars only; ``transform`` is a pure function of
    the supplied window + frozen scaler. ``warmup`` is the minimum window length
    required before ``transform`` yields a valid vector.
    """

    @property
    def warmup(self) -> int: ...

    @property
    def feature_names(self) -> list[str]: ...

    def fit(self, bars: Sequence[Bar]) -> None: ...

    def transform(self, window: Sequence[Bar]) -> FeatureVector: ...


class ManifoldModel(Protocol):
    """Embedding φ + regime assignment + neighbourhood forward-return model.

    Purely numeric. ``fit`` sees the TRAIN feature matrix and the realized forward
    returns aligned to each row (NaN for rows with no realized forward return).
    ``transform_online`` assigns a single new point WITHOUT re-fitting.
    """

    def fit(self, X: np.ndarray, fwd_returns: np.ndarray) -> None: ...

    def transform_online(self, x: np.ndarray) -> ManifoldState: ...

    @property
    def regimes(self) -> list[Regime]: ...


class Strategy(Protocol):
    """Turns manifold geometry into signals, then into a target exposure."""

    def signals(self, state: ManifoldState) -> SignalSet: ...

    def target(self, signals: SignalSet) -> TargetPosition: ...


class RiskSizer(Protocol):
    """Sizes a target weight into a concrete order given risk state.

    Returns ``None`` when no trade is required (already at target within band).
    """

    def size(
        self,
        target: TargetPosition,
        portfolio: PortfolioState,
        price: float,
        anomaly: float,
        volatility: float,
    ) -> Order | None: ...


class StateStore(Protocol):
    """Persists bars + run artifacts. Local DuckDB/Parquet is the reference impl;
    a Supabase exporter implements the same Protocol behind the scenes."""

    def append_bars(self, bars: Sequence[Bar]) -> None: ...

    def write_run(self, run_id: str, results: Sequence[StepResult]) -> None: ...

    def write_regimes(self, run_id: str, regimes: Sequence[Regime]) -> None: ...

    def query(self, sql: str) -> list[dict]: ...
