"""The two load-bearing correctness tests for the Mannofold engine.

1. Backtest ≡ Paper: the same series through the historical and live-replay feeds
   must produce bit-identical StepResults — proving the single online step.
2. No-lookahead: mutating future bars must not change any step computed before
   the mutation point — proving no future information leaks into a decision.
"""

from __future__ import annotations

from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.live_replay import LiveReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars

CFG = EngineConfig(train_size=400, refit_every=300, max_train=1500)


def _run(feed, run_id="t"):
    # Pin BLAS/OpenMP to one thread: KMeans' threaded float reductions are otherwise
    # non-deterministic at the ULP level, which would mask the real invariant.
    with threadpool_limits(limits=1):
        return Engine(config=CFG, run_id=run_id).run(feed).results


def test_backtest_equals_paper():
    bars, _ = generate_bars(SyntheticConfig(n_bars=1600, seed=11))
    bt = _run(HistoricalReplayFeed(bars), "bt")
    paper = _run(LiveReplayFeed(bars, speed=0.0), "paper")

    assert len(bt) == len(paper) and len(bt) > 0
    for a, b in zip(bt, paper, strict=True):
        assert a.seq == b.seq
        assert a.manifold.embedding == b.manifold.embedding
        assert a.manifold.regime_id == b.manifold.regime_id
        assert a.target.target_weight == b.target.target_weight
        assert a.portfolio.equity == b.portfolio.equity


def test_no_lookahead():
    bars, _ = generate_bars(SyntheticConfig(n_bars=1600, seed=11))
    divergence = 1200  # well into the online phase, after a refit boundary

    mutated = list(bars)
    for i in range(divergence, len(mutated)):
        b = mutated[i]
        mutated[i] = b.model_copy(update={"close": b.close * 1.5, "high": b.high * 1.5})

    base = _run(HistoricalReplayFeed(bars), "base")
    pert = _run(HistoricalReplayFeed(mutated), "pert")

    # Steps for bars strictly before the mutation point must be unchanged.
    cutoff_ts = bars[divergence].ts
    pre_base = [r for r in base if r.bar.ts < cutoff_ts]
    pre_pert = [r for r in pert if r.bar.ts < cutoff_ts]

    assert len(pre_base) == len(pre_pert) and len(pre_base) > 0
    for a, b in zip(pre_base, pre_pert, strict=True):
        assert a.model_dump() == b.model_dump(), f"lookahead leak at seq {a.seq}"
