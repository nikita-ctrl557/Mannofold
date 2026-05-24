"""The orchestrator: the only component that drives time forward.

``Engine.run(feed)`` is identical for backtest and paper — the feed and clock are
the only difference. It owns the walk-forward refit schedule and portfolio
accounting, and emits :class:`StreamEvent`s for the realtime API.
"""

from mannofold.engine.engine import Engine, EngineConfig, RunResult
from mannofold.engine.metrics import compute_metrics

__all__ = ["Engine", "EngineConfig", "RunResult", "compute_metrics"]
