"""LocalStateStore round-trips bars, run artifacts, and regimes under tmp_path."""

from __future__ import annotations

import json

from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore

CFG = EngineConfig(train_size=400, refit_every=300, max_train=1500)


def _make_run(store: LocalStateStore, run_id: str = "store_run"):
    bars, _ = generate_bars(SyntheticConfig(n_bars=1000, seed=9))
    feed = HistoricalReplayFeed(bars)
    with threadpool_limits(limits=1):
        return Engine(config=CFG, store=store, run_id=run_id).run(feed)


def test_run_writes_parseable_artifacts(tmp_path):
    store = LocalStateStore(tmp_path)
    run_id = "store_run"
    result = _make_run(store, run_id)

    run_dir = tmp_path / "runs" / run_id
    run_json = run_dir / "run.json"
    regimes_json = run_dir / "regimes.json"

    assert run_json.exists()
    assert regimes_json.exists()

    run_doc = json.loads(run_json.read_text())
    assert run_doc["run_id"] == run_id
    assert len(run_doc["steps"]) == len(result.results)

    regimes_doc = json.loads(regimes_json.read_text())
    assert isinstance(regimes_doc, list)
    assert len(regimes_doc) == len(result.regimes)


def test_query_counts_bars(tmp_path):
    store = LocalStateStore(tmp_path)
    _make_run(store)

    rows = store.query("SELECT count(*) AS n FROM bars")
    assert len(rows) == 1
    assert rows[0]["n"] > 0


def test_append_bars_then_query_symbol(tmp_path):
    store = LocalStateStore(tmp_path)
    bars, _ = generate_bars(SyntheticConfig(n_bars=300, seed=2))
    store.append_bars(bars)

    rows = store.query("SELECT count(*) AS n FROM bars")
    assert rows[0]["n"] == len(bars)

    symbols = store.query("SELECT DISTINCT symbol AS s FROM bars")
    assert {r["s"] for r in symbols} == {bars[0].symbol}


def test_append_empty_bars_is_noop(tmp_path):
    store = LocalStateStore(tmp_path)
    store.append_bars([])  # must not raise
    bars, _ = generate_bars(SyntheticConfig(n_bars=120, seed=4))
    store.append_bars(bars)
    rows = store.query("SELECT count(*) AS n FROM bars")
    assert rows[0]["n"] == len(bars)
