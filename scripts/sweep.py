"""Quick parameter sweep over the strategy + risk knobs.

Runs a handful of backtests on (smallish) synthetic data and prints a table
ranked by Sharpe. Each combo varies:

* strategy ``gain``        — conviction-to-weight slope;
* risk ``rebalance_band``  — churn / no-trade band;
* risk ``target_vol``      — vol-targeted gross exposure.

Reuses :class:`Engine` + :func:`compute_metrics` exactly as the backtest does;
threads are pinned to one so the runs are deterministic.

Usage: python scripts/sweep.py [n_bars]
"""

from __future__ import annotations

import sys
from itertools import product

from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.signals.risk import VolTargetRiskSizer
from mannofold.signals.strategy import ManifoldStrategy

GAINS = (40.0, 60.0, 90.0)
BANDS = (0.02, 0.05)
TARGET_VOLS = (0.008, 0.012)


def _run_one(bars, gain: float, band: float, target_vol: float) -> dict:
    cfg = EngineConfig(
        train_size=400,
        refit_every=300,
        max_train=1500,
        target_vol=target_vol,
        rebalance_band=band,
    )
    strategy = ManifoldStrategy(gain=gain)
    risk = VolTargetRiskSizer(
        target_vol=target_vol,
        max_leverage=cfg.max_leverage,
        rebalance_band=band,
        commission_bps=cfg.commission_bps,
    )
    with threadpool_limits(limits=1):
        result = Engine(
            config=cfg, strategy=strategy, risk=risk, run_id="sweep"
        ).run(HistoricalReplayFeed(bars))
    m = compute_metrics(result.results)
    m.update(gain=gain, band=band, target_vol=target_vol)
    return m


def main() -> None:
    n_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    bars, _ = generate_bars(SyntheticConfig(n_bars=n_bars, seed=7))

    rows = [
        _run_one(bars, gain, band, tv)
        for gain, band, tv in product(GAINS, BANDS, TARGET_VOLS)
    ]
    rows.sort(key=lambda r: r.get("sharpe", 0.0), reverse=True)

    header = (
        f"{'gain':>6} {'band':>6} {'tvol':>6} | "
        f"{'trades':>7} {'return%':>9} {'sharpe':>8} {'maxDD%':>8}"
    )
    print(f"sweep over {len(rows)} combos, n_bars={n_bars}\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['gain']:>6.0f} {r['band']:>6.3f} {r['target_vol']:>6.3f} | "
            f"{r['n_trades']:>7d} {r['total_return'] * 100:>9.2f} "
            f"{r['sharpe']:>8.3f} {r['max_drawdown'] * 100:>8.2f}"
        )

    best = rows[0]
    print(
        f"\nbest by sharpe: gain={best['gain']:.0f} band={best['band']:.3f} "
        f"tvol={best['target_vol']:.3f}  sharpe={best['sharpe']:.3f} "
        f"return={best['total_return'] * 100:.2f}% trades={best['n_trades']}"
    )


if __name__ == "__main__":
    main()
