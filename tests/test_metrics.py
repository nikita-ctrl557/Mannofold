"""compute_metrics produces every documented key with finite, sane values.

We drive a small real run through the engine on synthetic bars so the metrics are
computed from genuine StepResults rather than hand-built fixtures.
"""

from __future__ import annotations

import math

from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars

# Small config keeps the run fast while still exercising a refit boundary.
CFG = EngineConfig(train_size=400, refit_every=300, max_train=1500)

EXPECTED_KEYS = {
    "n_steps",
    "n_trades",
    "total_return",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "final_equity",
}


def _run_results():
    bars, _ = generate_bars(SyntheticConfig(n_bars=1200, seed=5))
    with threadpool_limits(limits=1):
        return Engine(config=CFG, run_id="metrics").run(HistoricalReplayFeed(bars)).results


def test_metrics_have_all_documented_keys():
    metrics = compute_metrics(_run_results())
    assert set(metrics.keys()) == EXPECTED_KEYS


def test_metrics_values_finite_and_sane():
    results = _run_results()
    metrics = compute_metrics(results)

    assert metrics["n_steps"] == len(results)
    assert metrics["n_steps"] > 0

    assert isinstance(metrics["n_trades"], int)
    assert 0 <= metrics["n_trades"] <= metrics["n_steps"]

    for key in ("total_return", "sharpe", "max_drawdown", "final_equity"):
        assert math.isfinite(metrics[key]), key

    assert metrics["final_equity"] > 0.0
    assert 0.0 <= metrics["win_rate"] <= 1.0
    # Drawdown is measured as equity/peak - 1, so it is never positive.
    assert metrics["max_drawdown"] <= 0.0


def test_empty_results_returns_empty_dict():
    assert compute_metrics([]) == {}
