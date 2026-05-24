"""Core domain models — pure pydantic, zero heavy dependencies.

These are the wire + storage + in-process types. The WebSocket event schema
(``events.py``) and the generated TypeScript types both derive from here, so
this module is the contract drift boundary between Python and the frontend.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# Sentinel regime id for "off-manifold" / anomalous states (HDBSCAN noise label).
ANOMALY_REGIME = -1


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


class Bar(BaseModel):
    """A single OHLCV bar. The atomic unit the engine steps over."""

    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureVector(BaseModel):
    """High-dimensional market-state vector x_t ∈ R^D built from a bar window."""

    ts: datetime
    symbol: str
    values: list[float]
    names: list[str] = Field(default_factory=list)


class Regime(BaseModel):
    """Metadata for a manifold regime (cluster), produced at fit time."""

    regime_id: int
    label: str = ""
    color: str = "#888888"
    size: int = 0
    mean_fwd_return: float = 0.0


class ManifoldState(BaseModel):
    """The geometric position + derived quantities for one market state.

    All fields are computed ONLINE from a frozen model — never from data with
    ``ts`` in the future of this bar.
    """

    ts: datetime
    symbol: str
    embedding: list[float]  # 2 or 3 dims — the visualization coordinate
    regime_id: int = ANOMALY_REGIME
    regime_prob: float = 0.0
    density: float = 0.0  # local neighbourhood density (higher = more typical)
    anomaly_score: float = 0.0  # distance-from-manifold (higher = more anomalous)
    fwd_return_mean: float = 0.0  # neighbourhood forward-return mean (train-only)
    fwd_return_std: float = 0.0
    velocity: list[float] = Field(default_factory=list)  # Δembedding over k steps


class SignalSet(BaseModel):
    """Trading signals derived from manifold geometry."""

    ts: datetime
    symbol: str
    momentum: float = 0.0  # trajectory velocity along return-gradient
    expected_return: float = 0.0  # from neighbourhood forward-return model
    anomaly: float = 0.0  # 0..1, drives de-grossing
    regime_id: int = ANOMALY_REGIME
    confidence: float = 0.0  # 0..1


class TargetPosition(BaseModel):
    """Desired exposure as a signed fraction of equity in [-1, 1]."""

    ts: datetime
    symbol: str
    target_weight: float = 0.0


class Order(BaseModel):
    ts: datetime
    symbol: str
    side: Side
    qty: float
    target_weight: float = 0.0
    reason: str = ""


class Fill(BaseModel):
    ts: datetime
    symbol: str
    side: Side
    qty: float
    price: float
    commission: float = 0.0


class PortfolioState(BaseModel):
    ts: datetime
    cash: float
    equity: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    positions: dict[str, float] = Field(default_factory=dict)  # symbol -> qty
    returns: float = 0.0  # period return
    drawdown: float = 0.0  # current drawdown from peak


class StepResult(BaseModel):
    """Everything produced by a single engine step — the unit of the event stream.

    The golden equivalence test asserts this is bit-identical between backtest and
    paper modes when replaying the same series.
    """

    seq: int
    mode: Mode
    bar: Bar
    features: FeatureVector
    manifold: ManifoldState
    signals: SignalSet
    target: TargetPosition
    order: Order | None = None
    fill: Fill | None = None
    portfolio: PortfolioState
