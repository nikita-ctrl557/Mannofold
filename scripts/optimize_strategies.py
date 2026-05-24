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


def _backtest(bars: list[Bar], strategy_build, tail: int | None = None) -> dict:
    """Run a backtest. When ``tail`` is set, the model trains on the leading
    prefix and metrics are measured only on the final ``tail`` steps — used by
    the short 3-month tests so a pre-trained model is evaluated on the window."""
    n = len(bars)
    cfg = EngineConfig(train_size=max(60, min(400, n // 3)), refit_every=250,
                       max_train=1000, target_vol=0.01, rebalance_band=0.03)
    risk = VolTargetRiskSizer(target_vol=0.01, max_leverage=cfg.max_leverage, rebalance_band=0.03)
    res = Engine(config=cfg, strategy=strategy_build(), risk=risk, run_id="opt").run(
        HistoricalReplayFeed(bars))
    results = res.results[-tail:] if tail else res.results
    m = compute_metrics(results, periods_per_year=252)
    m["yoy"] = _yoy(results)
    return m


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
            a["win_rate"] += m["win_rate"]; a["max_drawdown"] += m["max_drawdown"]
            a["n"] += 1
            by_scenario.setdefault(e.name, {})[sid] = {
                "total_return": round(m["total_return"], 4), "sharpe": round(m["sharpe"], 3),
                "win_rate": round(m["win_rate"], 4), "max_drawdown": round(m["max_drawdown"], 4),
                "n_trades": m["n_trades"], "yoy": m["yoy"],
            }
        winner = max(results_here, key=lambda kv: kv[1]["sharpe"])
        wins[winner[0]] += 1
        best_per_scenario[sid] = {"strategy": winner[0], "sharpe": round(winner[1]["sharpe"], 3)}
        print(f"{sclist['label']:34s} winner={winner[0]:18s} sharpe={winner[1]['sharpe']:+.2f}")

    strategies = []
    for e in entries:
        a = agg[e.name]; n = a["n"] or 1
        strategies.append({
            "name": e.name, "description": e.description,
            "mean_sharpe": round(a["sharpe"] / n, 3),
            "mean_return": round(a["total_return"] / n, 4),
            "mean_win_rate": round(a["win_rate"] / n, 4),
            "mean_max_drawdown": round(a["max_drawdown"] / n, 4),
            "scenario_wins": wins[e.name], "n_scenarios": int(n),
            "by_scenario": by_scenario[e.name],
        })
    strategies.sort(key=lambda s: s["mean_sharpe"], reverse=True)

    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "scenarios": [{k: s[k] for k in ("id", "label", "kind", "instrument", "note")}
                      for s in scenarios],
        "ranking": [s["name"] for s in strategies],
        "best_per_scenario": best_per_scenario,
        "strategies": strategies,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(strategies)} strategies over {len(scenarios)} scenarios -> {OUT}")
    print("ranking by mean Sharpe:", payload["ranking"])


if __name__ == "__main__":
    main()
