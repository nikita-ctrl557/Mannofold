"""Precompute real-data backtests and stage them as a static bundle for the web UI.

Runs a backtest per free GitHub dataset, then writes the run + regimes JSON under
``web/public/runs/<id>/`` plus an ``index.json`` manifest and ``datasets.json``.
The web app reads these as a static fallback when no live API is present, so the
dashboard renders real history (and lets you switch datasets) when hosted as a
pure static bundle — e.g. inside the Studio site at ``/manifold/``.

Usage: python scripts/build_web_runs.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mannofold.engine import Engine, EngineConfig
from mannofold.feed.github_csv import DATASETS, GithubCsvFeed
from mannofold.persist.store import LocalStateStore

# Manifest order = web default; compact + deepest-history dataset first so the
# mobile first paint is fast.
ORDER = ["sp500", "vix", "aapl"]

OUT = Path("web/public/runs")


def main() -> None:
    store = LocalStateStore()
    OUT.mkdir(parents=True, exist_ok=True)
    datasets_meta: list[dict] = []

    for name in ORDER:
        if name not in DATASETS:
            continue
        feed = GithubCsvFeed(name)
        start, end = feed.date_range()
        cfg = EngineConfig(
            train_size=min(500, len(feed) // 3), refit_every=250, max_train=1000
        )
        result = Engine(config=cfg, store=store, run_id=name).run(feed)
        src = Path(store.data_dir) / "runs" / name
        dst = OUT / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("run.json", "regimes.json"):
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
        datasets_meta.append(
            {
                "name": name,
                "symbol": DATASETS[name].symbol,
                "n_bars": len(feed),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "description": DATASETS[name].description,
            }
        )
        print(f"staged {name}: {len(result.results)} steps, {len(result.regimes)} regimes")

    (OUT / "index.json").write_text(json.dumps({"runs": ORDER}))
    (OUT / "datasets.json").write_text(json.dumps({"datasets": datasets_meta}))
    print(f"wrote manifest + datasets.json -> {OUT}")


if __name__ == "__main__":
    main()
