"""No-lookahead guarantee, enforced for EVERY registered strategy engine.

The engine feeds each strategy a causal stream of ManifoldStates (one per bar,
in order), so a strategy cannot read a future bar. This test proves it the hard
way for all 37+ engines: mutate every bar at/after a divergence point and assert
that every StepResult produced for an earlier bar is bit-identical. Any future
information leaking into an earlier decision would change those steps.
"""

from __future__ import annotations

import pytest
from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.signals.strategies import discover

CFG = EngineConfig(train_size=300, refit_every=200, max_train=900)
ENGINES = discover()


def _run(bars, build, run_id):
    with threadpool_limits(limits=1):
        return Engine(config=CFG, strategy=build(), run_id=run_id).run(
            HistoricalReplayFeed(bars)
        ).results


@pytest.mark.parametrize("entry", ENGINES, ids=[e.name for e in ENGINES])
def test_engine_is_blind(entry):
    bars, _ = generate_bars(SyntheticConfig(n_bars=900, seed=11))
    divergence = 650  # well into the online phase

    mutated = list(bars)
    for i in range(divergence, len(mutated)):
        b = mutated[i]
        mutated[i] = b.model_copy(update={"close": b.close * 1.5, "high": b.high * 1.5})

    base = _run(bars, entry.build, "base")
    pert = _run(mutated, entry.build, "pert")

    cutoff_ts = bars[divergence].ts
    pre_base = [r for r in base if r.bar.ts < cutoff_ts]
    pre_pert = [r for r in pert if r.bar.ts < cutoff_ts]

    assert len(pre_base) == len(pre_pert) and len(pre_base) > 0
    for a, b in zip(pre_base, pre_pert, strict=True):
        assert a.model_dump() == b.model_dump(), (
            f"{entry.name}: lookahead leak at seq {a.seq}"
        )
