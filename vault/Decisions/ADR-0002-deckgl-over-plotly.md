---
title: ADR-0002 — deck.gl over Plotly
tags: [adr, decision, frontend]
---

# ADR-0002 — deck.gl (WebGL) for the manifold, not Plotly

Back to [[Home]] · Related: [[Architecture]] · [[Manifold Embedding]] ·
[[ADR-0001-stack]] · [[Glossary]]

Status: **Accepted**

## Context

The manifold map is the centrepiece visualization: a scatter of **embedded market
states** (`ManifoldState.embedding`, [[Manifold Embedding]]) coloured by regime,
plus a trajectory trail. A live run streams steps continuously and the frontend
retains up to `LIVE_CAP = 50000` points ([[Architecture]]). SVG/DOM-based charting
(Plotly, D3, Recharts) creates a node per point and collapses well before that
scale; we need tens of thousands of points panning/zooming at 60fps.

## Decision

Render the manifold with **deck.gl** (`@deck.gl/core`, `@deck.gl/layers`,
`@deck.gl/react`, v9), GPU-accelerated via WebGL. `web/src/components/ManifoldMap.tsx`
uses a `ScatterplotLayer` (regime-coloured, radius scaled by `density`, pickable
for hover tooltips) + a `PathLayer` trail of the last `TRAIL = 50` states, inside
an `OrthographicView` (the embedding is an abstract 2-D plane, not a geo map).

Time-series panels (equity/drawdown, signals, feature heatmap) use **uPlot**
(`web/src/components/UPlotChart.tsx`) — a tiny, fast canvas charting library —
rather than Plotly, for the same performance-at-scale reason.

## Consequences

- Smooth interaction at the target point counts; the visualization *is* the
  product, so this is load-bearing.
- Two focused rendering libraries (deck.gl for the point cloud, uPlot for series)
  instead of one heavyweight do-everything chart lib.
- Slightly more glue code (manual view-bounds fitting, RGB color mapping,
  `requestAnimationFrame` batching of live steps) versus Plotly's batteries-included
  API — an accepted trade for performance.
