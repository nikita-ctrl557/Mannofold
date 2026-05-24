# Mannofold

A **market-manifold trading engine** with a high-density visualization.

Mannofold treats each market state as a high-dimensional feature vector `x_t ∈ R^D`
(multi-horizon returns, realized vol, momentum, RSI, volume features) and posits that
market states live near a low-dimensional **manifold**. It learns an embedding
`φ: R^D → R^{2,3}`, clusters the manifold into **regimes**, and derives trading
signals from the geometry: where you sit on the manifold (neighbourhood forward
returns), how fast you're moving (trajectory velocity), and how far you are from the
manifold (anomaly → de-gross). The embedding *is* the visualization.

One **strategy core** serves both **backtest** (historical replay) and **paper**
(live/replay feed) — they differ only in the data feed and clock, which is enforced
by a golden equivalence test.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
uv run python scripts/gen_synthetic.py        # zero-secret synthetic dataset
uv run python scripts/run_backtest.py demo    # run a backtest, persist artifacts
uv run pytest -q                              # backtest≡paper + no-lookahead tests
```

No API keys or external data are needed: a regime-switching synthetic generator makes
the whole stack runnable offline. Live data via Alpaca (paper) is an optional drop-in
feed (`MANNOFOLD_ALPACA_KEY` / `MANNOFOLD_ALPACA_SECRET` env vars — never committed).

## Layout

| Path | What |
|------|------|
| `mannofold/contracts/` | Frozen pydantic models + Protocol interfaces (the source of truth) |
| `mannofold/feed/` | Data feeds: synthetic, historical replay, live replay, Alpaca |
| `mannofold/features/` | Rolling feature pipeline (scaler embedded → no lookahead) |
| `mannofold/manifold/` | Swappable embedding φ (PCA→UMAP) + regimes + forward-return model |
| `mannofold/signals/` | Manifold-geometry strategy + vol-targeting risk sizer |
| `mannofold/engine/` | The single online step; walk-forward refit; portfolio accounting |
| `mannofold/persist/` | DuckDB + partitioned Parquet store |
| `mannofold/api/` | FastAPI + WebSocket realtime stream |
| `web/` | React + TypeScript dashboard (deck.gl manifold + uPlot charts) |
| `vault/` | Obsidian knowledge vault: theory, architecture, decisions |

See [`vault/Home.md`](vault/Home.md) for the full theory and architecture notes.
