"""Portfolio-level ("hedge fund") backtest + professional performance analytics.

Runs ONE strategy through the single unified engine across a whole universe of
tickers, equal-weights the per-name strategy returns into one daily-rebalanced
book, and reports the metrics a desk actually reviews: CAGR, annualized vol,
Sharpe, Sortino, Calmar, max drawdown, hit rate, plus alpha / beta / information
ratio / correlation versus an equal-weight buy-and-hold benchmark of the same
universe.

No lookahead is inherited from the engine (each name is a causal walk-forward
run); the portfolio layer only aggregates realized per-bar returns by date.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import Bar
from mannofold.engine.engine import Engine, EngineConfig
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.signals.risk import VolTargetRiskSizer

_TRADING_DAYS = 252


def pro_metrics(rets: list[float], rf_annual: float = 0.02) -> dict:
    """Professional return-series metrics on a daily return series."""
    n = len(rets)
    if n == 0:
        return {k: 0.0 for k in (
            "total_return", "cagr", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "hit_rate", "best_day", "worst_day")} | {"n_days": 0}
    mean = sum(rets) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / n)
    rf = rf_annual / _TRADING_DAYS
    em = mean - rf
    sharpe = (em / std) * math.sqrt(_TRADING_DAYS) if std > 0 else 0.0
    dstd = math.sqrt(sum(min(0.0, r - rf) ** 2 for r in rets) / n)
    sortino = (em / dstd) * math.sqrt(_TRADING_DAYS) if dstd > 0 else 0.0
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for r in rets:
        eq *= 1 + r
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1)
    years = n / _TRADING_DAYS
    cagr = eq ** (1 / years) - 1 if years > 0 and eq > 0 else 0.0
    return {
        "n_days": n,
        "total_return": round(eq - 1, 4),
        "cagr": round(cagr, 4),
        "ann_vol": round(std * math.sqrt(_TRADING_DAYS), 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(maxdd, 4),
        "calmar": round(cagr / abs(maxdd), 3) if maxdd < 0 else 0.0,
        "hit_rate": round(sum(1 for r in rets if r > 0) / n, 4),
        "best_day": round(max(rets), 4),
        "worst_day": round(min(rets), 4),
    }


def _vs_benchmark(port: list[float], bench: list[float]) -> dict:
    """Beta / annualized alpha / information ratio / correlation vs benchmark."""
    n = min(len(port), len(bench))
    if n < 2:
        return {"beta": 0.0, "alpha_annual": 0.0, "info_ratio": 0.0, "correlation": 0.0}
    p = port[-n:]
    b = bench[-n:]
    mp = sum(p) / n
    mb = sum(b) / n
    cov = sum((p[i] - mp) * (b[i] - mb) for i in range(n)) / n
    vb = sum((b[i] - mb) ** 2 for i in range(n)) / n
    vp = sum((p[i] - mp) ** 2 for i in range(n)) / n
    beta = cov / vb if vb > 0 else 0.0
    alpha_daily = mp - beta * mb
    diff = [p[i] - b[i] for i in range(n)]
    md = sum(diff) / n
    sd = math.sqrt(sum((d - md) ** 2 for d in diff) / n)
    corr = cov / math.sqrt(vp * vb) if vp > 0 and vb > 0 else 0.0
    return {
        "beta": round(beta, 3),
        "alpha_annual": round(alpha_daily * _TRADING_DAYS, 4),
        "info_ratio": round((md / sd) * math.sqrt(_TRADING_DAYS), 3) if sd > 0 else 0.0,
        "correlation": round(corr, 3),
    }


def run_portfolio(
    strategy_build: Callable[[], Strategy],
    universe: dict[str, list[Bar]],
    target_vol: float = 0.02,
) -> dict:
    """Equal-weight, daily-rebalanced portfolio of the strategy run per ticker."""
    cfg = EngineConfig(train_size=400, refit_every=250, max_train=1000,
                       target_vol=target_vol, rebalance_band=0.03)
    strat_by_day: dict[str, list[float]] = defaultdict(list)
    bench_by_day: dict[str, list[float]] = defaultdict(list)
    n_names = 0
    for ticker, bars in universe.items():
        if len(bars) < 450:
            continue
        risk = VolTargetRiskSizer(target_vol=target_vol, max_leverage=cfg.max_leverage,
                                  rebalance_band=0.03)
        res = Engine(config=cfg, strategy=strategy_build(), risk=risk,
                     run_id="pf").run(HistoricalReplayFeed(bars))
        if not res.results:
            continue
        n_names += 1
        for sr in res.results:
            strat_by_day[sr.bar.ts.date().isoformat()].append(sr.portfolio.returns)
        for i in range(1, len(bars)):
            prev, cur = bars[i - 1].close, bars[i].close
            if prev > 0:
                bench_by_day[bars[i].ts.date().isoformat()].append(cur / prev - 1.0)

    days = sorted(strat_by_day)
    port = [sum(v) / len(v) for v in (strat_by_day[d] for d in days) if v]
    bench_days = sorted(bench_by_day)
    bench = [sum(v) / len(v) for v in (bench_by_day[d] for d in bench_days) if v]

    m = pro_metrics(port)
    bm = pro_metrics(bench)
    rel = _vs_benchmark(port, bench)
    return {
        "n_names": n_names,
        "portfolio": m,
        "benchmark": bm,
        "vs_benchmark": rel,
    }
