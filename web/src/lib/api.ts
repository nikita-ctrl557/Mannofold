import type { RunData, Regime, Metrics, StepResult } from "../types/contracts";

const FALLBACK_REGIMES: Regime[] = [];

async function tryJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export interface LoadResult {
  run: RunData;
  regimes: Regime[];
  source: "api" | "sample";
}

// List available runs from the API (empty when offline).
export async function listRuns(): Promise<string[]> {
  const list = await tryJson<{ runs: string[] }>("/api/runs");
  return list?.runs ?? [];
}

export interface DatasetInfo {
  name: string;
  symbol: string;
  n_bars: number | null;
  start: string | null;
  end: string | null;
  description: string;
}

// Datasets available for the simulation view (synthetic + free historical).
export async function listDatasets(): Promise<DatasetInfo[]> {
  const r = await tryJson<{ datasets: DatasetInfo[] }>("/api/datasets");
  return r?.datasets ?? [];
}

// Try the live API first; fall back to bundled sample for offline dev.
export async function loadRun(runId?: string): Promise<LoadResult> {
  let id = runId;
  if (!id) {
    const list = await tryJson<{ runs: string[] }>("/api/runs");
    if (list && list.runs && list.runs.length) id = list.runs[0];
  }
  if (id) {
    const run = await tryJson<RunData>(`/api/runs/${id}`);
    if (run && run.steps && run.steps.length) {
      const regimes =
        (await tryJson<Regime[]>(`/api/runs/${id}/regimes`)) ??
        deriveRegimes(run.steps);
      return { run, regimes, source: "api" };
    }
  }
  // offline path — bundled samples live next to index.html, so resolve them
  // against the Vite base (BASE_URL) to stay correct under a hosted subpath.
  const base = import.meta.env.BASE_URL;
  const run = await tryJson<RunData>(`${base}sample-run.json`);
  if (!run) {
    return {
      run: { run_id: "empty", steps: [] },
      regimes: FALLBACK_REGIMES,
      source: "sample",
    };
  }
  const regimes =
    (await tryJson<Regime[]>(`${base}sample-regimes.json`)) ??
    deriveRegimes(run.steps);
  return { run, regimes, source: "sample" };
}

export async function loadMetrics(
  id: string,
  steps: StepResult[]
): Promise<Metrics> {
  const m = await tryJson<Metrics>(`/api/runs/${id}/metrics`);
  return m ?? computeMetrics(steps);
}

const PALETTE = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
];

// Build regime metadata from steps when the server doesn't supply it.
export function deriveRegimes(steps: StepResult[]): Regime[] {
  const byId = new Map<number, { count: number; fwdSum: number }>();
  for (const s of steps) {
    const rid = s.manifold.regime_id;
    if (rid < 0) continue;
    const e = byId.get(rid) ?? { count: 0, fwdSum: 0 };
    e.count += 1;
    e.fwdSum += s.manifold.fwd_return_mean;
    byId.set(rid, e);
  }
  return [...byId.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([rid, e]) => ({
      regime_id: rid,
      label: `regime ${rid}`,
      color: PALETTE[rid % PALETTE.length],
      size: e.count,
      mean_fwd_return: e.count ? e.fwdSum / e.count : 0,
    }));
}

// Local metrics fallback mirroring the /metrics endpoint shape.
export function computeMetrics(steps: StepResult[]): Metrics {
  if (!steps.length) {
    return {
      n_steps: 0,
      n_trades: 0,
      total_return: 0,
      sharpe: 0,
      max_drawdown: 0,
      win_rate: 0,
      final_equity: 0,
    };
  }
  const eq = steps.map((s) => s.portfolio.equity);
  const e0 = eq[0];
  const eN = eq[eq.length - 1];
  const rets = steps.map((s) => s.portfolio.returns);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance =
    rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / rets.length;
  const std = Math.sqrt(variance) || 1e-9;
  const sharpe = (mean / std) * Math.sqrt(252 * 24); // hourly bars annualized
  let maxDd = 0;
  for (const s of steps) maxDd = Math.min(maxDd, s.portfolio.drawdown);
  const fills = steps.filter((s) => s.fill != null);
  const wins = rets.filter((r) => r > 0).length;
  return {
    n_steps: steps.length,
    n_trades: fills.length,
    total_return: e0 ? eN / e0 - 1 : 0,
    sharpe,
    max_drawdown: maxDd,
    win_rate: rets.length ? wins / rets.length : 0,
    final_equity: eN,
  };
}

export function regimeColorMap(regimes: Regime[]): Map<number, string> {
  const m = new Map<number, string>();
  for (const r of regimes) m.set(r.regime_id, r.color);
  return m;
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const v =
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h;
  const n = parseInt(v, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
