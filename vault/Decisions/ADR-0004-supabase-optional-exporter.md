---
title: ADR-0004 — Supabase as optional exporter
tags: [adr, decision, persistence]
---

# ADR-0004 — Local-first store; Supabase as an optional Protocol exporter

Back to [[Home]] · Related: [[Architecture]] · [[ADR-0005-synthetic-data-zero-secrets]] ·
[[ADR-0001-stack]] · [[Glossary]]

Status: **Accepted**

## Context

The Mannofold container is **ephemeral**. The engine needs durable, queryable run
artifacts on its critical path, but it must also run with **zero secrets**
([[ADR-0005-synthetic-data-zero-secrets]]). A shared Postgres (Supabase) is useful
for a long-lived frontend to read finished runs *after* the container is gone, but
it must never be load-bearing or require credentials to operate.

## Decision

- The reference `StateStore` is **`LocalStateStore`** (`mannofold/persist/store.py`):
  partitioned micro-batch **Parquet** + a **DuckDB** glob view + `run.json` /
  `regimes.json` snapshots ([[Architecture]]). This is the only store on the
  engine's inline write path.
- **`SupabaseExporter`** (`mannofold/persist/supabase_export.py`) implements the
  *same* `StateStore` Protocol and ships a finished run's artifacts to Supabase
  Postgres over the **PostgREST** REST API (`httpx`, no extra deps). It is wired
  in as an optional *secondary* fan-out at run end.
- It is built to **never be load-bearing**: with no `SUPABASE_URL` /
  `SUPABASE_KEY` it constructs fine and every method no-ops after one
  informational message; network failures during export are caught and logged,
  never raised. Its expected table schema is documented in the module docstring.

## Consequences

- Local DuckDB/Parquet remains the source of truth; cross-store SQL stays local
  (`SupabaseExporter.query` returns `[]` rather than pretending to be a SQL
  endpoint).
- A failed or absent export can never take down a run — the zero-secret default
  path is fully functional.
- Scale-out (shared read-after-run access) is available by setting two env vars,
  with no change to the engine.
