"""Tests for the analytics layer + the optional Supabase exporter.

The analytics test runs a small real Engine over synthetic bars (kept tiny for
speed) and asserts the shape/keys of ``analyze`` plus the per-regime count
reconciliation. The exporter test asserts import-safe, no-op construction with no
credentials present.
"""

from __future__ import annotations

import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from mannofold.analytics import (
    analyze,
    drawdown_series,
    exposure_stats,
    per_regime_pnl,
    return_distribution,
    rolling_sharpe,
    turnover,
)
from mannofold.engine import Engine, EngineConfig
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore
from mannofold.persist.supabase_export import SupabaseExporter

# Small config: enough bars to fit + take a chunk of online steps, but fast.
CFG = EngineConfig(train_size=300, refit_every=400, max_train=1500, n_regimes=3)


@pytest.fixture(scope="module")
def run_results():
    bars, _ = generate_bars(SyntheticConfig(n_bars=700, seed=5))
    with threadpool_limits(limits=1):
        rr = Engine(config=CFG, run_id="analytics").run(HistoricalReplayFeed(bars))
    assert len(rr.results) > 0
    return rr.results


def test_analyze_keys_and_shapes(run_results):
    out = analyze(run_results)
    expected = {
        "n_steps",
        "rolling_sharpe",
        "per_regime_pnl",
        "turnover",
        "exposure_stats",
        "drawdown_series",
        "return_distribution",
    }
    assert expected <= set(out)
    assert out["n_steps"] == len(run_results)
    # Series-valued analytics are one record per step.
    assert len(out["rolling_sharpe"]) == len(run_results)
    assert len(out["drawdown_series"]) == len(run_results)


def test_per_regime_counts_reconcile(run_results):
    reg = per_regime_pnl(run_results)
    assert set(["regime_id", "count", "pnl", "mean_pnl", "hit_rate"]) <= set(reg.columns)
    # Reconciliation: every trading step is attributed to exactly one regime.
    assert int(reg["count"].sum()) == len(run_results)
    assert (reg["hit_rate"] >= 0).all() and (reg["hit_rate"] <= 1).all()


def test_individual_analytics_shapes(run_results):
    rs = rolling_sharpe(run_results, window=16)
    assert list(rs.columns) == ["seq", "rolling_sharpe"]
    assert len(rs) == len(run_results)

    dd = drawdown_series(run_results)
    assert set(["seq", "equity", "peak", "drawdown"]) <= set(dd.columns)
    # Drawdown is never positive; peak is monotonic non-decreasing.
    assert (dd["drawdown"] <= 1e-9).all()
    assert (dd["peak"].diff().fillna(0.0) >= -1e-9).all()

    expo = exposure_stats(run_results)
    assert {"avg_gross", "max_gross", "avg_net", "max_net"} <= set(expo)
    assert expo["max_gross"] >= expo["avg_gross"]

    to = turnover(run_results)
    assert {"total", "avg_per_step", "max_step", "n_trades"} <= set(to)
    assert to["total"] >= 0.0

    rd = return_distribution(run_results)
    assert {"mean", "std", "skew", "kurtosis", "var", "cvar", "n"} <= set(rd)
    assert rd["n"] == len(run_results)
    assert rd["std"] >= 0.0


def test_analyze_from_dataframe_matches(tmp_path, run_results):
    """Analytics off the on-disk flattened frame reconcile the same way."""
    store = LocalStateStore(data_dir=tmp_path)
    store.write_run("analytics", run_results)
    df = store.load_steps_df("analytics")
    assert isinstance(df, pd.DataFrame) and len(df) == len(run_results)

    reg = per_regime_pnl(df)
    assert int(reg["count"].sum()) == len(run_results)
    out = analyze(df)
    assert out["n_steps"] == len(run_results)


def test_empty_input():
    assert analyze([])["n_steps"] == 0
    assert return_distribution([])["n"] == 0
    assert per_regime_pnl([]).empty


def test_supabase_exporter_noops_without_creds(monkeypatch, capsys):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    exp = SupabaseExporter()
    assert exp.enabled is False

    bars, _ = generate_bars(SyntheticConfig(n_bars=10, seed=1))
    # None of these may raise or attempt network I/O.
    exp.append_bars(bars)
    exp.write_run("r", [])
    exp.write_regimes("r", [])
    assert exp.query("SELECT 1") == []

    # The one-time notice was emitted at construction.
    assert "no-op" in capsys.readouterr().out.lower()


def test_supabase_exporter_satisfies_protocol(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    # StateStore is a (non-runtime) Protocol, so verify conformance structurally:
    # every required method exists and is callable.
    exp = SupabaseExporter()
    for method in ("append_bars", "write_run", "write_regimes", "query"):
        assert callable(getattr(exp, method))
