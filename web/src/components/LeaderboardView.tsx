import { useEffect, useMemo, useState } from "react";

interface ScenarioResult {
  total_return: number;
  sharpe: number;
  win_rate: number;
  max_drawdown: number;
  n_trades: number;
  yoy: { year: number; return: number }[];
}
interface StrategyRow {
  name: string;
  description: string;
  mean_sharpe: number;
  mean_return: number;
  mean_win_rate: number;
  mean_max_drawdown: number;
  scenario_wins: number;
  n_scenarios: number;
  by_scenario: Record<string, ScenarioResult>;
}
interface Scenario {
  id: string;
  label: string;
  kind: "real" | "synthetic";
  instrument: string;
  note: string;
}
interface Leaderboard {
  generated_at: string;
  scenarios: Scenario[];
  ranking: string[];
  best_per_scenario: Record<string, { strategy: string; sharpe: number }>;
  strategies: StrategyRow[];
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;
const cls = (x: number) => (x > 0 ? "good" : x < 0 ? "bad" : "");

async function load(): Promise<Leaderboard | null> {
  const base = import.meta.env.BASE_URL;
  for (const url of ["/api/leaderboard", `${base}runs/leaderboard.json`]) {
    try {
      const r = await fetch(url);
      if (r.ok) return (await r.json()) as Leaderboard;
    } catch {
      /* try next */
    }
  }
  return null;
}

export default function LeaderboardView() {
  const [lb, setLb] = useState<Leaderboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    load().then((d) => {
      if (!alive) return;
      setLb(d);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, []);

  const scenarioById = useMemo(() => {
    const m = new Map<string, Scenario>();
    lb?.scenarios.forEach((s) => m.set(s.id, s));
    return m;
  }, [lb]);

  if (loading) return <div className="lb-empty">computing leaderboard…</div>;
  if (!lb || !lb.strategies?.length)
    return <div className="lb-empty">no leaderboard yet — run scripts/optimize_strategies.py</div>;

  return (
    <div className="lb">
      <div className="lb-head">
        <div>
          <div className="lb-title">strategy leaderboard</div>
          <div className="lb-sub">
            {lb.strategies.length} strategies · {lb.scenarios.length} backtests each ·
            ranked by mean Sharpe across all scenarios
          </div>
        </div>
        <div className="lb-gen">
          generated {new Date(lb.generated_at).toLocaleString()}
        </div>
      </div>

      <div className="lb-table">
        <div className="lb-row lb-header">
          <span>#</span>
          <span>strategy</span>
          <span>mean&nbsp;Sharpe</span>
          <span>mean&nbsp;return</span>
          <span>win&nbsp;rate</span>
          <span>mean&nbsp;maxDD</span>
          <span>#&nbsp;wins</span>
        </div>
        {lb.strategies.map((s, i) => (
          <div key={s.name}>
            <div
              className="lb-row lb-click"
              onClick={() => setOpen(open === s.name ? null : s.name)}
            >
              <span className="lb-rank">{i + 1}</span>
              <span className="lb-name">
                {s.name}
                <span className="lb-desc">{s.description}</span>
              </span>
              <span className="lb-num" data-label="Sharpe">{s.mean_sharpe.toFixed(2)}</span>
              <span className={`lb-num ${cls(s.mean_return)}`} data-label="Return">{signed(s.mean_return)}</span>
              <span className="lb-num" data-label="Win">{pct(s.mean_win_rate)}</span>
              <span className="lb-num bad" data-label="MaxDD">{pct(s.mean_max_drawdown)}</span>
              <span className="lb-num" data-label="Wins">{s.scenario_wins}</span>
            </div>
            {open === s.name && (
              <div className="lb-detail">
                {Object.entries(s.by_scenario).map(([sid, r]) => {
                  const sc = scenarioById.get(sid);
                  return (
                    <div key={sid} className="lb-scn">
                      <span className="lb-scn-label">
                        {sc?.label ?? sid}
                        <span className={`lb-tag ${sc?.kind ?? ""}`}>{sc?.kind}</span>
                      </span>
                      <span className={`lb-num ${cls(r.total_return)}`}>{signed(r.total_return)}</span>
                      <span className="lb-num">Sh {r.sharpe.toFixed(2)}</span>
                      <span className="lb-num">{pct(r.win_rate)} win</span>
                      <span className="lb-num">{r.n_trades} trades</span>
                      {r.yoy?.length > 1 && (
                        <span className="lb-yoy">
                          {r.yoy.map((y) => (
                            <span key={y.year} className={`lb-yoy-cell ${cls(y.return)}`} title={`${y.year}: ${signed(y.return)}`}>
                              {String(y.year).slice(2)} {signed(y.return)}
                            </span>
                          ))}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="lb-best">
        <div className="lb-best-title">best engine per scenario — "which engine where"</div>
        <div className="lb-best-grid">
          {lb.scenarios.map((sc) => {
            const b = lb.best_per_scenario[sc.id];
            return (
              <div key={sc.id} className="lb-best-cell">
                <span className="lb-best-scn">{sc.label}</span>
                <span className="lb-best-strat">{b?.strategy ?? "—"}</span>
                <span className="lb-best-note">{sc.note}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
