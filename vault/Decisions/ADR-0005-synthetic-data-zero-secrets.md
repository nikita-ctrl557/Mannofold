---
title: ADR-0005 — Synthetic data, zero secrets
tags: [adr, decision, data]
---

# ADR-0005 — Synthetic regime-switching data; runnable offline with zero secrets

Back to [[Home]] · Related: [[Regime Detection]] · [[Architecture]] ·
[[ADR-0006-alpaca-paper-feed]] · [[ADR-0004-supabase-optional-exporter]] ·
[[Glossary]]

Status: **Accepted**

## Context

Mannofold must be **runnable in an ephemeral container with no API keys and no
external data** — for CI, for review, and to keep the default path free of
secrets. It also needs **ground-truth regime labels** to validate that the
manifold pipeline actually rediscovers regimes ([[Regime Detection]]), which real
market data cannot provide.

## Decision

Ship a deterministic **regime-switching synthetic generator**
(`mannofold/feed/synthetic.py`): a 3-state Markov-switching geometric Brownian
motion —

- state 0 — low-vol trending (positive drift, low vol),
- state 1 — high-vol mean-reverting (zero drift, high vol),
- state 2 — crash (strong negative drift, very high vol, rare/short).

`generate_bars(SyntheticConfig)` returns `(bars, regime_labels)` where the labels
are the **true** hidden state per bar. A fixed `seed` makes runs reproducible.
This is the default data source for the engine, the API (`POST /api/runs` accepts
`n_bars` / `seed`), and the correctness tests.

## Consequences

- The whole stack runs offline with one command, no credentials — supports the
  zero-secret invariant and CI.
- The known labels enable a regime-recovery check and give the manifold something
  with genuine structure (clusters + a rare crash mode → off-manifold anomalies,
  [[Mannofold Theory]]) to learn.
- Live market data is strictly **opt-in** via `AlpacaFeed`
  ([[ADR-0006-alpaca-paper-feed]]); any external persistence is opt-in too
  ([[ADR-0004-supabase-optional-exporter]]). Secrets, when used, come from env
  vars and are never committed or logged.
- Synthetic ≠ real-market microstructure; it validates correctness and pipeline
  mechanics, not alpha. That is the intended scope.
