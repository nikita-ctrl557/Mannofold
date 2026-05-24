---
title: ADR-0001 — Stack & package layout
tags: [adr, decision]
---

# ADR-0001 — Language, libraries, and package layout

Back to [[Home]] · Related: [[Architecture]] · [[ADR-0002-deckgl-over-plotly]] ·
[[ADR-0003-pca-baseline-umap-swappable]] · [[Glossary]]

Status: **Accepted**

## Context

Mannofold must run end-to-end in an ephemeral container with **zero secrets**
([[ADR-0005-synthetic-data-zero-secrets]]), be parallelisable across independent
workstreams, and keep a clean Python↔frontend contract boundary. The numerical
core (features, PCA/kNN/KMeans) is squarely in the scientific-Python ecosystem.

## Decision

- **Python ≥ 3.11** for the engine (`requires-python = ">=3.11"`,
  `pyproject.toml`); **TypeScript + React** for the dashboard.
- Core deps (always installable): `numpy`, `pandas`, `scikit-learn`, `scipy`,
  `duckdb`, `pyarrow`, `pydantic` v2, `fastapi`, `uvicorn`, `httpx`,
  `websockets`. Heavy compiled manifold libs (`umap-learn`, `hdbscan`) are an
  **opt-in `manifold` extra** so the baseline always installs cleanly
  ([[ADR-0003-pca-baseline-umap-swappable]]).
- `uv` for env + locking (`uv.lock`); `hatchling` build backend.
- Tooling: `ruff` (lint + import sort), `mypy` with the pydantic plugin,
  `pytest`. Configured in `pyproject.toml`.
- **Package layout** mirrors the dependency graph ([[Architecture]]): every
  capability is its own subpackage (`feed/`, `features/`, `manifold/`,
  `signals/`, `engine/`, `persist/`, `api/`) depending only on
  `contracts/` (frozen pydantic **models** + **Protocol** interfaces). The
  frontend lives in `web/`; docs in `vault/`.

## Consequences

- Protocols (`mannofold/contracts/interfaces.py`) are the seams the parallel
  workstreams build against — implementations stay swappable and independently
  testable.
- pydantic models double as wire, storage, and in-process types, and export a
  JSON-Schema that generates the TS types — Python and frontend cannot drift
  ([[Architecture]]).
- The baseline install is lightweight; UMAP/HDBSCAN are added only when wanted.
- Mature, well-supported libraries; no exotic dependencies on the hot path.
