"""Richer analytics over a run's StepResults.

These functions COMPLEMENT :func:`mannofold.engine.metrics.compute_metrics`
(headline scalars) with vectorized, distribution- and regime-aware views. They
are pure: given a sequence of StepResults (or a flattened steps DataFrame) they
return plain dicts / DataFrames and never mutate inputs or touch I/O.
"""

from mannofold.analytics.metrics import (
    analyze,
    drawdown_series,
    exposure_stats,
    per_regime_pnl,
    return_distribution,
    rolling_sharpe,
    turnover,
)

__all__ = [
    "analyze",
    "drawdown_series",
    "exposure_stats",
    "per_regime_pnl",
    "return_distribution",
    "rolling_sharpe",
    "turnover",
]
