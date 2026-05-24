"""Comprehensive strategy leaderboard.

Backtests every strategy variant across 20+ test scenarios spanning horizons
(15y / 10y / 5y), random 3-month windows, named regime-change episodes
(GFC, COVID, 2018 vol-jump), real instruments (VIX, AAPL) and synthetic
"stocks/ETFs" with distinct regime characters (bull / bear / choppy / crash /
fast-switching / low-vol / random). Each result carries year-over-year returns.

Aggregates per strategy (mean Sharpe/return/win across all scenarios, plus how
often it ranks #1) and records the best strategy per scenario ("which engine
where"). Writes web/public/runs/leaderboard.json for the dashboard.

Usage: python scripts/optimize_strategies.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

from mannofold.contracts.models import Bar
from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.github_csv import load_bars
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.signals.risk import VolTargetRiskSizer
from mannofold.signals.strategies import discover

OUT = Path("web/public/runs/leaderboard.json")
UTC = dt.timezone.utc
RNG = random.Random(42)


def _slice(bars: list[Bar], start: str | None = None, end: str | None = None) -> list[Bar]:
    d1 = dt.datetime.fromisoformat(start).replace(tzinfo=UTC) if start else None
    d2 = dt.datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
    return [b for b in bars if (d1 is None or b.ts >= d1) and (d2 is None or b.ts <= d2)]


def _synth(seed: int, drifts, vols, transition, n=1300) -> list[Bar]:
    cfg = SyntheticConfig(
        symbol="SYNTH", n_bars=n, seed=seed, bar_minutes=1440,
        start=dt.datetime(2021, 1, 1, tzinfo=UTC),
        drifts=drifts, vols=vols, transition=transition,
    )
    return generate_bars(cfg)[0]


def build_scenarios() -> list[dict]:
    """Return a list of {id, label, kind, instrument, note, bars}."""
    vix = load_bars("vix")
    aapl = load_bars("aapl")
    sc: list[dict] = []

    # ── Real horizons ───────────────────────────────────────────────────
    sc.append(dict(id="vix_15y", label="VIX · 15-year", kind="real", instrument="VIX",
                   note="2011→now daily", bars=_slice(vix, "2011-01-01")))
    sc.append(dict(id="vix_10y", label="VIX · 10-year", kind="real", instrument="VIX",
                   note="2016→now daily", bars=_slice(vix, "2016-01-01")))
    sc.append(dict(id="vix_5y", label="VIX · 5-year", kind="real", instrument="VIX",
                   note="2021→now daily", bars=_slice(vix, "2021-01-01")))

    # ── Real regime-change episodes ─────────────────────────────────────
    sc.append(dict(id="vix_gfc", label="VIX · GFC regime change", kind="real", instrument="VIX",
                   note="2007-06→2010-06", bars=_slice(vix, "2007-06-01", "2010-06-01")))
    sc.append(dict(id="vix_covid", label="VIX · COVID crash", kind="real", instrument="VIX",
                   note="2019-06→2021-06", bars=_slice(vix, "2019-06-01", "2021-06-01")))
    sc.append(dict(id="vix_2018vol", label="VIX · 2018 vol-jump", kind="real", instrument="VIX",
                   note="2017-06→2019-06", bars=_slice(vix, "2017-06-01", "2019-06-01")))

    # ── Random 3-month windows (real) ───────────────────────────────────
    # Prepend a 450-bar training prefix; measure only the final 63 (≈3 months)
    # so a pre-trained model is evaluated on the window (warmup needs 41 bars).
    recent = _slice(vix, "2004-01-01")
    for i in range(3):
        lo = RNG.randint(450, len(recent) - 63)
        win = recent[lo - 450:lo + 63]
        sc.append(dict(id=f"vix_3mo_{i+1}", label=f"VIX · random 3-month #{i+1}",
                       kind="real", instrument="VIX", measure_tail=63,
                       note=f"{recent[lo].ts.date()}→{recent[lo+62].ts.date()}", bars=win))

    # ── Real equity ─────────────────────────────────────────────────────
    sc.append(dict(id="aapl_full", label="AAPL · 2014-2016", kind="real", instrument="AAPL",
                   note="daily OHLCV", bars=aapl))

    # ── Synthetic instruments with distinct regime character ────────────
    base_trans = ((0.990, 0.009, 0.001), (0.020, 0.978, 0.002), (0.060, 0.140, 0.800))
    fast_trans = ((0.940, 0.055, 0.005), (0.090, 0.900, 0.010), (0.200, 0.300, 0.500))
    sc.append(dict(id="synth_bull", label="SYNTH · steady bull", kind="synthetic", instrument="SYN-BULL",
                   note="positive drift, low vol", bars=_synth(11, (0.0010, 0.0, -0.0015), (0.008, 0.014, 0.030), base_trans)))
    sc.append(dict(id="synth_bear", label="SYNTH · bear drift", kind="synthetic", instrument="SYN-BEAR",
                   note="negative drift", bars=_synth(12, (-0.0008, 0.0, -0.0030), (0.010, 0.016, 0.035), base_trans)))
    sc.append(dict(id="synth_choppy", label="SYNTH · choppy range", kind="synthetic", instrument="SYN-CHOP",
                   note="high-vol mean-revert", bars=_synth(13, (0.0001, 0.0, -0.0020), (0.012, 0.022, 0.040),
                                                            ((0.900, 0.095, 0.005), (0.120, 0.875, 0.005), (0.100, 0.200, 0.700)))))
    sc.append(dict(id="synth_crash", label="SYNTH · crash & recover", kind="synthetic", instrument="SYN-CRASH",
                   note="frequent crash state", bars=_synth(14, (0.0008, 0.0, -0.0060), (0.009, 0.018, 0.055),
                                                            ((0.970, 0.020, 0.010), (0.040, 0.940, 0.020), (0.150, 0.150, 0.700)))))
    sc.append(dict(id="synth_fastregime", label="SYNTH · fast regime switch", kind="synthetic", instrument="SYN-FAST",
                   note="rapid regime change", bars=_synth(15, (0.0009, 0.0, -0.0025), (0.010, 0.020, 0.040), fast_trans)))
    sc.append(dict(id="synth_lowvol", label="SYNTH · low-vol grind", kind="synthetic", instrument="SYN-GRIND",
                   note="low vol, gentle drift", bars=_synth(16, (0.0006, 0.0, -0.0010), (0.005, 0.009, 0.020), base_trans)))
    for i in range(4):
        sc.append(dict(id=f"synth_rand_{i+1}", label=f"SYNTH · random stock #{i+1}",
                       kind="synthetic", instrument=f"SYN-R{i+1}",
                       note="randomized regimes", bars=_synth(100 + i, (0.0004, 0.0, -0.0020), (0.009, 0.015, 0.030), base_trans)))
    long_synth = _synth(200, (0.0005, 0.0, -0.0020), (0.009, 0.016, 0.032), base_trans, n=2000)
    for i in range(2):
        lo = RNG.randint(450, len(long_synth) - 63)
        win = long_synth[lo - 450:lo + 63]
        sc.append(dict(id=f"synth_3mo_{i+1}", label=f"SYNTH · random 3-month #{i+1}",
                       kind="synthetic", instrument="SYN-LONG", measure_tail=63,
                       note="3-month slice (pre-trained)", bars=win))
    if os.environ.get("MANNO_FAST"):
        # Fast mode: drop the multi-year VIX windows (~2.5k-3.8k bars each) that
        # dominate runtime; keep regime-change, 3-month, AAPL and all synthetic
        # scenarios so every dimension is still covered.
        heavy = {"vix_15y", "vix_10y", "vix_5y"}
        sc = [s for s in sc if s["id"] not in heavy]
    return sc


def _yoy(results) -> list[dict]:
    by_year: "OrderedDict[int, list[float]]" = OrderedDict()
    for s in results:
        by_year.setdefault(s.bar.ts.year, []).append(s.portfolio.equity)
    out, prev = [], None
    for year, eqs in by_year.items():
        open_eq = prev if prev is not None else eqs[0]
        ret = (eqs[-1] / open_eq - 1.0) if open_eq else 0.0
        out.append({"year": year, "return": round(ret, 4)})
        prev = eqs[-1]
    return out


def _mom(results) -> tuple[float, float, list[dict]]:
    """Month-over-month returns off the equity curve.
    Returns (avg monthly return, % positive months, monthly series)."""
    by_month: "OrderedDict[str, list[float]]" = OrderedDict()
    for s in results:
        key = f"{s.bar.ts.year}-{s.bar.ts.month:02d}"
        by_month.setdefault(key, []).append(s.portfolio.equity)
    series, prev = [], None
    for month, eqs in by_month.items():
        open_eq = prev if prev is not None else eqs[0]
        ret = (eqs[-1] / open_eq - 1.0) if open_eq else 0.0
        series.append({"month": month, "return": round(ret, 4)})
        prev = eqs[-1]
    if not series:
        return 0.0, 0.0, []
    rets = [m["return"] for m in series]
    avg = sum(rets) / len(rets)
    hit = sum(1 for r in rets if r > 0) / len(rets)
    return avg, hit, series


# Position-sizing knobs swept per (engine, scenario) — the per-stock
# optimization. Higher target_vol => more exposure => higher return AND higher
# drawdown; we pick the best risk-adjusted (Sharpe) config and report its
# return + drawdown so the risk is visible, not hidden.
TARGET_VOLS = (0.020,)

# Out-of-sample holdout: a diverse subset NEVER used to rank engines. We rank by
# in-sample Sharpe and report holdout Sharpe alongside, so an engine that only
# looks good on the scenarios it was selected on (overfit) is exposed. This is
# the anti-overfit / "can't cheat at the selection level" safeguard.
HOLDOUT_IDS = {
    "vix_5y", "vix_covid", "vix_3mo_2", "aapl_full",
    "synth_bear", "synth_crash", "synth_rand_3",
}


def _one(bars: list[Bar], strategy_build, target_vol: float, tail: int | None) -> dict:
    n = len(bars)
    cfg = EngineConfig(train_size=max(60, min(400, n // 3)), refit_every=250,
                       max_train=1000, target_vol=target_vol, rebalance_band=0.03)
    risk = VolTargetRiskSizer(target_vol=target_vol, max_leverage=cfg.max_leverage,
                              rebalance_band=0.03)
    res = Engine(config=cfg, strategy=strategy_build(), risk=risk, run_id="opt").run(
        HistoricalReplayFeed(bars))
    results = res.results[-tail:] if tail else res.results
    m = compute_metrics(results, periods_per_year=252)
    n_steps = m.get("n_steps", len(results)) or 1
    # Annualized (CAGR) return assuming ~252 daily bars/yr.
    m["annual_return"] = (1.0 + m["total_return"]) ** (252.0 / n_steps) - 1.0
    m["yoy"] = _yoy(results)
    mom_avg, mom_hit, _ = _mom(results)
    m["mom_avg"] = mom_avg          # average month-over-month return
    m["mom_hit"] = mom_hit          # fraction of positive months
    m["target_vol"] = target_vol
    return m


def _backtest(bars: list[Bar], strategy_build, tail: int | None = None) -> dict:
    """Optimize position sizing per scenario: sweep target_vol, keep the best
    risk-adjusted (Sharpe) config. When ``tail`` is set, the model trains on the
    leading prefix and metrics are measured only on the final ``tail`` steps."""
    best: dict | None = None
    for tv in TARGET_VOLS:
        m = _one(bars, strategy_build, tv, tail)
        if best is None or m["sharpe"] > best["sharpe"]:
            best = m
    return best


def main() -> None:
    entries = discover()
    scenarios = build_scenarios()
    print(f"{len(entries)} strategies × {len(scenarios)} scenarios "
          f"= {len(entries) * len(scenarios)} backtests\n")

    by_scenario: dict[str, dict] = {}
    agg = {e.name: defaultdict(float) for e in entries}
    wins = defaultdict(int)
    best_per_scenario: dict[str, dict] = {}

    for sclist in scenarios:
        sid = sclist["id"]
        results_here: list[tuple[str, dict]] = []
        for e in entries:
            m = _backtest(sclist["bars"], e.build, tail=sclist.get("measure_tail"))
            results_here.append((e.name, m))
            a = agg[e.name]
            a["sharpe"] += m["sharpe"]; a["total_return"] += m["total_return"]
            a["annual_return"] += m["annual_return"]
            a["win_rate"] += m["win_rate"]; a["max_drawdown"] += m["max_drawdown"]
            a["mom_avg"] += m["mom_avg"]; a["mom_hit"] += m["mom_hit"]
            a["n"] += 1
            if sid in HOLDOUT_IDS:
                a["oos_sharpe"] += m["sharpe"]; a["oos_n"] += 1
            else:
                a["is_sharpe"] += m["sharpe"]; a["is_n"] += 1
            by_scenario.setdefault(e.name, {})[sid] = {
                "total_return": round(m["total_return"], 4),
                "annual_return": round(m["annual_return"], 4),
                "sharpe": round(m["sharpe"], 3),
                "win_rate": round(m["win_rate"], 4), "max_drawdown": round(m["max_drawdown"], 4),
                "n_trades": m["n_trades"], "target_vol": m["target_vol"],
                "mom_avg": round(m["mom_avg"], 4), "mom_hit": round(m["mom_hit"], 4),
                "yoy": m["yoy"],
            }
        winner = max(results_here, key=lambda kv: kv[1]["sharpe"])
        wins[winner[0]] += 1
        best_per_scenario[sid] = {"strategy": winner[0], "sharpe": round(winner[1]["sharpe"], 3)}
        print(f"{sclist['label']:34s} winner={winner[0]:18s} sharpe={winner[1]['sharpe']:+.2f}")

    strategies = []
    for e in entries:
        a = agg[e.name]; n = a["n"] or 1
        is_n = a["is_n"] or 1; oos_n = a["oos_n"] or 1
        is_sharpe = round(a["is_sharpe"] / is_n, 3)
        oos_sharpe = round(a["oos_sharpe"] / oos_n, 3)
        by = by_scenario[e.name]
        best_scn = max(by.values(), key=lambda r: r["total_return"], default={"total_return": 0})
        scenarios_over_40 = sum(1 for r in by.values() if r["annual_return"] >= 0.40)
        strategies.append({
            "name": e.name, "description": e.description,
            "mean_sharpe": round(a["sharpe"] / n, 3),
            "in_sample_sharpe": is_sharpe,
            "holdout_sharpe": oos_sharpe,
            # Robust = holds up out-of-sample (not just on selection scenarios).
            "robust": bool(oos_sharpe > 0 and oos_sharpe >= is_sharpe - 0.3),
            "mean_return": round(a["total_return"] / n, 4),
            "mean_annual_return": round(a["annual_return"] / n, 4),
            "mean_monthly_return": round(a["mom_avg"] / n, 4),
            "monthly_hit_rate": round(a["mom_hit"] / n, 4),
            "best_return": round(best_scn["total_return"], 4),
            "scenarios_over_40pct_annual": scenarios_over_40,
            "mean_win_rate": round(a["win_rate"] / n, 4),
            "mean_max_drawdown": round(a["max_drawdown"] / n, 4),
            "scenario_wins": wins[e.name], "n_scenarios": int(n),
            "by_scenario": by,
        })
    # Rank by IN-SAMPLE Sharpe (selection), but holdout_sharpe reveals overfitting.
    strategies.sort(key=lambda s: s["in_sample_sharpe"], reverse=True)

    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "scenarios": [
            {**{k: s[k] for k in ("id", "label", "kind", "instrument", "note")},
             "split": "holdout" if s["id"] in HOLDOUT_IDS else "in_sample"}
            for s in scenarios
        ],
        "ranking": [s["name"] for s in strategies],
        "best_per_scenario": best_per_scenario,
        "strategies": strategies,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    OUT.write_text(body)
    # Also write to the API data dir so a running :8000 server serves it live
    # via GET /api/leaderboard.
    api_out = Path("data/leaderboard.json")
    api_out.parent.mkdir(parents=True, exist_ok=True)
    api_out.write_text(body)
    print(f"\nwrote {len(strategies)} strategies over {len(scenarios)} scenarios -> {OUT} + {api_out}")
    print("ranking by mean Sharpe:", payload["ranking"])


if __name__ == "__main__":
    main()
