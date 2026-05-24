---
title: Home
tags: [moc, index]
---

# Mannofold — Project Brain

**Mannofold** is a market-manifold trading engine with a high-density
visualization. It treats each market state as a high-dimensional feature vector
`x_t ∈ R^D`, posits those states live near a low-dimensional **manifold**, learns
an embedding `φ: R^D → R^{2,3}`, clusters the manifold into **regimes**, and
derives trading signals from the geometry. A single strategy core serves both
**backtest** and **paper**, enforced by a golden equivalence test.

> Source of truth: this vault is documentation. The code under `mannofold/` is
> authoritative; every note here cites real file paths.

This note is the **Map of Content (MOC)**. Every other note links back here and
to its neighbours, so the Obsidian graph view shows the project as a connected
whole.

## Theory

- [[Mannofold Theory]] — the market-manifold thesis, the 11 features, the three
  signal mechanisms.
- [[Manifold Embedding]] — PCA → UMAP → Parametric UMAP, transform stability,
  fit-on-train-only.
- [[Regime Detection]] — KMeans vs HDBSCAN, `approximate_predict`, the `-1`
  anomaly regime.
- [[Signal Geometry]] — how `expected_return` / `momentum` / `anomaly` /
  `confidence` are derived.
- [[Risk Model]] — vol targeting, anomaly de-grossing, the rebalance band.

## Architecture

- [[Architecture]] — component + dependency graph, the feed≡backtest/paper
  invariant, walk-forward refit, persistence, API/WS fan-out, frontend.
- [[No-Lookahead]] — point-in-time correctness rules and the tests that enforce
  them.

## Decisions (ADRs)

- [[ADR-0001-stack]] — language, libraries, package layout.
- [[ADR-0002-deckgl-over-plotly]] — WebGL manifold rendering.
- [[ADR-0003-pca-baseline-umap-swappable]] — PCA baseline, UMAP behind a factory.
- [[ADR-0004-supabase-optional-exporter]] — local-first store, Supabase as an
  optional Protocol exporter.
- [[ADR-0005-synthetic-data-zero-secrets]] — runnable offline, no secrets.
- [[ADR-0006-alpaca-paper-feed]] — optional live paper feed.

## Reference

- [[Glossary]] — manifold, embedding, regime, anomaly score, walk-forward,
  lookahead bias, vol targeting, and more.

## Quick map of the code

| Layer | Module | Note |
|-------|--------|------|
| Contracts | `mannofold/contracts/models.py`, `interfaces.py`, `events.py` | [[Architecture]] |
| Feed | `mannofold/feed/` | [[ADR-0006-alpaca-paper-feed]], [[ADR-0005-synthetic-data-zero-secrets]] |
| Features | `mannofold/features/pipeline.py`, `indicators.py` | [[Mannofold Theory]], [[No-Lookahead]] |
| Manifold | `mannofold/manifold/embedding.py`, `model.py` | [[Manifold Embedding]], [[Regime Detection]] |
| Signals | `mannofold/signals/strategy.py`, `risk.py` | [[Signal Geometry]], [[Risk Model]] |
| Engine | `mannofold/engine/engine.py` | [[Architecture]], [[No-Lookahead]] |
| Persist | `mannofold/persist/store.py` | [[ADR-0004-supabase-optional-exporter]] |
| API | `mannofold/api/app.py` | [[Architecture]] |
| Web | `web/src/` | [[ADR-0002-deckgl-over-plotly]] |
| Tests | `tests/test_correctness.py` | [[No-Lookahead]] |
