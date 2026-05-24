---
title: Architecture
tags: [architecture, design]
---

# Architecture — components, dependencies, data flow

Back to [[Home]] · Related: [[No-Lookahead]] · [[Mannofold Theory]] ·
[[Signal Geometry]] · [[Risk Model]] · [[Manifold Embedding]] ·
[[Regime Detection]] · [[ADR-0001-stack]] · [[ADR-0002-deckgl-over-plotly]] ·
[[ADR-0004-supabase-optional-exporter]] · [[Glossary]]

## Dependency graph

Everything is built against **Protocols + models**, not concrete modules. The
engine wires concrete implementations together but imports them only at
construction.

```text
                     contracts/  (models.py · interfaces.py · events.py)
                          │  (frozen pydantic types + Protocols — source of truth)
        ┌─────────────┬───┴────────┬──────────────┬───────────────┐
        ▼             ▼            ▼              ▼               ▼
      feed/       features/     manifold/      signals/       persist/
   DataFeed    FeaturePipeline ManifoldModel  Strategy        StateStore
 synthetic     pipeline.py     embedding.py   strategy.py     store.py (DuckDB/Parquet)
 historical    indicators.py   model.py       risk.py         supabase_export.py
 live_replay                   _umap.py       (RiskSizer)
 alpaca
        └─────────────┴────────────┴──────────────┴───────────────┘
                          │
                          ▼
                     engine/engine.py   ── the single online step ──┐
                          │                                          │
            ┌─────────────┴─────────────┐                           │
            ▼                           ▼                           ▼
        api/app.py                 persist/store.py            engine/metrics.py
   (FastAPI REST + WS)        (Parquet parts + run.json)      (analytics/metrics.py)
            │
            ▼
        web/  (React + TypeScript: deck.gl manifold + uPlot charts)
```

The **numeric boundary**: `FeaturePipeline` emits domain `FeatureVector`s; the
engine flattens `.values` into a numpy matrix for `ManifoldModel` (purely
numeric, ts/symbol-agnostic) and re-attaches `ts`/`symbol` to the returned
`ManifoldState`. See the docstring in `mannofold/contracts/interfaces.py`.

## Contracts as the drift boundary

`mannofold/contracts/models.py` holds the frozen pydantic types (`Bar`,
`FeatureVector`, `ManifoldState`, `SignalSet`, `TargetPosition`, `Order`, `Fill`,
`PortfolioState`, `StepResult`, `Regime`). `interfaces.py` holds the Protocols
(`DataFeed`, `FeaturePipeline`, `ManifoldModel`, `Strategy`, `RiskSizer`,
`StateStore`). `events.py` defines the WS `StreamEvent` envelope and exports a
JSON-Schema (`web/src/types/contracts.schema.json`) that generates the TS types —
so Python and the frontend cannot silently drift.

## The DataFeed ≡ backtest/paper invariant

Backtest and paper differ **only in the `DataFeed`** (`mannofold/feed/`):

- `HistoricalReplayFeed` (`mode = BACKTEST`) — yields bars as fast as possible.
- `LiveReplayFeed` (`mode = PAPER`) — yields the *identical* bars on a wall-clock
  cadence (`speed` seconds between bars); the engine never reads the wall clock,
  so `speed` cannot affect output.
- `AlpacaFeed` (`mode = PAPER`) — optional live OHLCV via Alpaca v2 REST
  ([[ADR-0006-alpaca-paper-feed]]).

Because the engine consumes only the bar stream, the same `online_step` produces
**bit-identical** `StepResult`s in both modes — asserted by
`tests/test_correctness.py::test_backtest_equals_paper` ([[No-Lookahead]]).

## Walk-forward refit

`engine.run()` runs three phases (`mannofold/engine/engine.py`):

1. **Accumulate** `train_size` bars (default 800) — no trading.
2. **Fit** scaler → φ → regimes → kNN forward-return model on the TRAIN slice
   only (`_fit` → `build_training`), capped at `max_train` (1500) bars.
3. **Online**: for each subsequent bar run one inference step and trade; refit on
   the trailing window every `refit_every` (400) bars. Regime ids are stable only
   within one fitted model, so they are re-emitted via a `regime_fit` event at
   each refit boundary ([[Regime Detection]], [[Manifold Embedding]]).

Every step appends a `StepResult` (bar, features, manifold, signals, target,
order, fill, portfolio) and emits a `step` event via the `on_event` hook.

## Persistence model

`mannofold/persist/store.py` (`LocalStateStore`) is the reference `StateStore`:

- **Bars**: immutable **micro-batch Parquet parts** under
  `data/bars/symbol=.../part-<uuid>.parquet`, queried through a DuckDB glob view.
  Never stream row-by-row into one Parquet file (metadata lives at the file tail;
  DuckDB is single-writer).
- **Runs**: at run end, `steps.parquet` (flattened rows) + `run.json` (full
  fidelity JSON snapshot for the frontend) + `regimes.json` under
  `data/runs/{run_id}/`.

`SupabaseExporter` implements the *same* Protocol as an optional scale-out
exporter, never on the engine's hot path — [[ADR-0004-supabase-optional-exporter]].

## API + WebSocket fan-out

`mannofold/api/app.py` (FastAPI). REST control plane: `POST /api/runs` starts a
run in an executor thread; `GET /api/runs`, `/api/runs/{id}`,
`/api/runs/{id}/regimes`, `/api/runs/{id}/metrics` read persisted artifacts.

The **load-bearing** part is the bridge between the **synchronous** engine
(running in a worker thread) and the **async** WebSocket fan-out:

- Each run gets a `Hub` (pub/sub registry). Each WS client owns a **bounded**
  `asyncio.Queue` (`maxsize = 1000`).
- The engine thread publishes via `loop.call_soon_threadsafe` — the only
  thread-safe way to touch asyncio primitives across threads.
- On a full client queue the hub **drops the oldest** event *for that client
  only* (the monotonic `seq` lets the client detect the gap). A slow client can
  never stall the engine; publish is O(n_subscribers) enqueue-or-drop.
- Late subscribers join mid-run with no history replay; a finished run's hub is
  closed and new subscribers are dismissed promptly.

## Frontend

`web/` — React + TypeScript (Vite), two rendering libraries:

- **deck.gl** (`@deck.gl/core|layers|react`, WebGL) renders the **manifold map**
  (`web/src/components/ManifoldMap.tsx`): a `ScatterplotLayer` of embedded states
  coloured by regime + a `PathLayer` trail of the recent trajectory, in an
  `OrthographicView`. Chosen for tens of thousands of points at 60fps —
  [[ADR-0002-deckgl-over-plotly]].
- **uPlot** (`web/src/components/UPlotChart.tsx`) powers the equity/drawdown,
  signals, and feature panels — a tiny, fast canvas charting lib.

Data arrives two ways: offline `GET /api/runs/{id}` (or bundled
`sample-run.json`), or live over the WS via `web/src/lib/stream.ts`, which
batches steps into a `requestAnimationFrame` flush and caps retained points
(`LIVE_CAP = 50000`). TS types derive from the contracts JSON-Schema
(`web/src/types/contracts.ts`).
