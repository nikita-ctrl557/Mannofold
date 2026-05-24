---
title: ADR-0006 — Alpaca paper feed
tags: [adr, decision, feed]
---

# ADR-0006 — Alpaca as an optional live paper-trading feed

Back to [[Home]] · Related: [[Architecture]] · [[ADR-0005-synthetic-data-zero-secrets]] ·
[[No-Lookahead]] · [[Glossary]]

Status: **Accepted**

## Context

The default data path is the zero-secret synthetic generator
([[ADR-0005-synthetic-data-zero-secrets]]). To demonstrate the system on real
prices we want a live feed — but it must be a drop-in that respects the
DataFeed≡backtest/paper invariant ([[Architecture]]), needs no secrets to import,
and never weakens the [[No-Lookahead]] guarantees.

## Decision

Add **`AlpacaFeed`** (`mannofold/feed/alpaca.py`), implementing the `DataFeed`
Protocol with `mode == Mode.PAPER`:

- Fetches historical OHLCV bars from the **Alpaca Market Data v2 REST API**,
  paginating via `next_page_token`, and yields domain `Bar` objects in
  chronological order — exactly the contract the engine consumes, so it slots in
  beside `HistoricalReplayFeed` / `LiveReplayFeed` with no engine change.
- Credentials come from env vars `MANNOFOLD_ALPACA_KEY` /
  `MANNOFOLD_ALPACA_SECRET` (paper keys start with `PK`), sent as request headers
  and **never logged or committed**. Missing creds raise a clear error only at
  stream time.
- Accepts an **injectable `httpx.Client`** so tests drive it with an
  `httpx.MockTransport` and never touch the network. Blocked egress raises a
  descriptive `ConnectionError` ("allowlist `*.alpaca.markets` or run locally").
- A real-time WebSocket `live_stream()` is documented as a stub
  (`NotImplementedError`) for a future build.

## Consequences

- Real data is available without disturbing the default offline path or the
  golden-equivalence test ([[No-Lookahead]]).
- The injected-client design keeps the adapter fully unit-testable offline; the
  sandbox blocks `*.alpaca.markets`, so live use needs egress or a local run.
- Yielding bars in time order and never yielding a future bar early preserves
  point-in-time correctness through the same engine step.
