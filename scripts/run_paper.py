"""Run Mannofold in PAPER mode over synthetic data via the live-replay feed.

The paper path is the *same* engine as the backtest — only the feed differs
(:class:`LiveReplayFeed` yields on a wall-clock cadence). ``speed`` controls the
seconds slept between bars; the default of ``0.0`` replays as fast as possible so
this is runnable in CI. Periodic progress is printed via an event hook, and the
final :func:`compute_metrics` summary is printed and persisted under run id
``paper`` through :class:`LocalStateStore`.

Usage: python scripts/run_paper.py [run_id] [speed]
"""

from __future__ import annotations

import json
import sys

from mannofold.contracts.events import StreamEvent
from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.live_replay import LiveReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore

PROGRESS_EVERY = 500


def _make_progress_hook():
    state = {"steps": 0}

    def hook(event: StreamEvent) -> None:
        if event.type == "step":
            state["steps"] += 1
            if state["steps"] % PROGRESS_EVERY == 0:
                payload = event.payload or {}
                pf = payload.get("portfolio", {})
                equity = pf.get("equity")
                eq = f"{equity:,.2f}" if isinstance(equity, (int, float)) else "n/a"
                print(f"  [paper] step {state['steps']:>5}  equity={eq}")
        elif event.type == "run_start":
            print(f"  [paper] run_start  mode={(event.payload or {}).get('mode')}")
        elif event.type == "run_end":
            print(f"  [paper] run_end    n_steps={(event.payload or {}).get('n_steps')}")

    return hook


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "paper"
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    bars, _ = generate_bars(SyntheticConfig(n_bars=4000, seed=7))
    feed = LiveReplayFeed(bars, speed=speed)
    engine = Engine(
        config=EngineConfig(),
        store=LocalStateStore(),
        on_event=_make_progress_hook(),
        run_id=run_id,
    )

    print(f"Running PAPER mode: run_id={run_id} speed={speed} bars={len(bars)}")
    result = engine.run(feed)
    metrics = compute_metrics(result.results)
    print(f"\nrun_id={run_id}  steps={len(result.results)}  regimes={len(result.regimes)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
