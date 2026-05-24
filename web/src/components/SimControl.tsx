import type { DatasetInfo } from "../lib/api";

export const SPEEDS: Record<string, number> = {
  ultra: 0,
  fast: 0.004,
  normal: 0.02,
  slow: 0.08,
};

const WINDOWS = [250, 500, 1000, 2000, 4000, 8000];

interface Props {
  datasets: DatasetInfo[];
  dataset: string;
  setDataset: (s: string) => void;
  window: number;
  setWindow: (n: number) => void;
  startFrac: number;
  setStartFrac: (f: number) => void;
  speed: string;
  setSpeed: (s: string) => void;
  startIdx: number;
  totalBars: number;
  info: DatasetInfo | null;
  live: boolean;
  stepCount: number;
  expected: number;
  onRun: () => void;
  onStop: () => void;
}

function dateAt(info: DatasetInfo | null, idx: number, n: number): string {
  if (!info?.start || !info?.end || !n) return "";
  const t0 = Date.parse(info.start);
  const t1 = Date.parse(info.end);
  return new Date(t0 + (idx / n) * (t1 - t0)).toISOString().slice(0, 10);
}

export default function SimControl(p: Props) {
  const isSynth = p.dataset === "synthetic";
  const windows = isSynth ? WINDOWS : WINDOWS.filter((w) => w <= p.totalBars);
  const fromLabel = dateAt(p.info, p.startIdx, p.totalBars);
  const toLabel = dateAt(p.info, p.startIdx + p.window, p.totalBars);
  const pctDone = p.expected ? Math.min(100, (p.stepCount / p.expected) * 100) : 0;

  return (
    <div className="simbar">
      <span className="sim-title">SIMULATE</span>

      <label className="sim-field">
        data
        <select
          value={p.dataset}
          disabled={p.live}
          onChange={(e) => p.setDataset(e.target.value)}
        >
          {p.datasets.map((d) => (
            <option key={d.name} value={d.name}>
              {d.symbol}
              {d.n_bars ? ` (${d.n_bars})` : ""}
            </option>
          ))}
        </select>
      </label>

      <label className="sim-field">
        {isSynth ? "bars" : "window"}
        <select
          value={p.window}
          disabled={p.live}
          onChange={(e) => p.setWindow(Number(e.target.value))}
        >
          {windows.map((w) => (
            <option key={w} value={w}>
              {w}
            </option>
          ))}
        </select>
      </label>

      {!isSynth && (
        <label className="sim-field grow">
          history start {fromLabel && <span className="sim-date">{fromLabel} → {toLabel}</span>}
          <input
            type="range"
            min={0}
            max={1000}
            value={Math.round(p.startFrac * 1000)}
            disabled={p.live}
            onChange={(e) => p.setStartFrac(Number(e.target.value) / 1000)}
          />
        </label>
      )}

      <div className="sim-field">
        speed
        <div className="seg">
          {Object.keys(SPEEDS).map((k) => (
            <button
              key={k}
              className={"seg-btn" + (p.speed === k ? " on" : "")}
              disabled={p.live}
              onClick={() => p.setSpeed(k)}
            >
              {k}
            </button>
          ))}
        </div>
      </div>

      {p.live ? (
        <button className="sim-run stop" onClick={p.onStop}>
          ■ STOP
        </button>
      ) : (
        <button className="sim-run" onClick={p.onRun}>
          ▶ RUN
        </button>
      )}

      <div className="sim-progress">
        <div className="sim-progress-bar" style={{ width: `${pctDone}%` }} />
        <span className="sim-progress-txt">
          {p.live || p.stepCount ? `${p.stepCount.toLocaleString()} steps` : "idle"}
        </span>
      </div>
    </div>
  );
}
