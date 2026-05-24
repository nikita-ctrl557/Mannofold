import type { Metrics, Regime } from "../types/contracts";

interface Props {
  metrics: Metrics | null;
  currentRegime: number;
  netExposure: number;
  regimes: Regime[];
  source: "api" | "sample" | null;
  live: boolean;
  status: string;
  onToggleLive: () => void;
}

function pct(v: number): string {
  return (v * 100).toFixed(2) + "%";
}

export default function Header({
  metrics,
  currentRegime,
  netExposure,
  regimes,
  source,
  live,
  status,
  onToggleLive,
}: Props) {
  const regLabel =
    regimes.find((r) => r.regime_id === currentRegime)?.label ?? "--";
  const regColor =
    regimes.find((r) => r.regime_id === currentRegime)?.color ?? "#666";

  return (
    <div className="header">
      <div className="brand">
        MANNO<span className="dot">·</span>FOLD
      </div>
      <div className="kpis">
        <Kpi
          label="total return"
          value={metrics ? pct(metrics.total_return) : "--"}
          tone={metrics ? (metrics.total_return >= 0 ? "good" : "bad") : ""}
        />
        <Kpi
          label="sharpe"
          value={metrics ? metrics.sharpe.toFixed(2) : "--"}
          tone={metrics ? (metrics.sharpe >= 0 ? "good" : "bad") : ""}
        />
        <Kpi
          label="max drawdown"
          value={metrics ? pct(metrics.max_drawdown) : "--"}
          tone="bad"
        />
        <Kpi
          label="win rate"
          value={metrics ? pct(metrics.win_rate) : "--"}
          tone=""
        />
        <Kpi
          label="trades"
          value={metrics ? String(metrics.n_trades) : "--"}
          tone=""
        />
        <Kpi
          label="regime"
          value={
            <>
              <span
                className="swatch"
                style={{ background: regColor }}
              />
              {currentRegime} {regLabel}
            </>
          }
          tone=""
        />
        <Kpi
          label="net exposure"
          value={pct(netExposure)}
          tone={netExposure >= 0 ? "good" : "bad"}
        />
      </div>
      <div className="live-ctl">
        {source && <span className="src-tag">{source}</span>}
        <span className="status">{status}</span>
        <button
          className={"live-btn" + (live ? " on" : "")}
          onClick={onToggleLive}
        >
          {live ? "● LIVE" : "○ LIVE"}
        </button>
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  tone: string;
}) {
  return (
    <div className="kpi">
      <span className="label">{label}</span>
      <span className={"val " + tone}>{value}</span>
    </div>
  );
}
