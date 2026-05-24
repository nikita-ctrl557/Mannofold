import type { Regime } from "../types/contracts";

interface Props {
  regimes: Regime[];
  currentRegime: number;
}

// Per-regime stats with color swatch; highlights the live regime row.
export default function RegimeLegend({ regimes, currentRegime }: Props) {
  if (!regimes.length) return <div className="empty">no regimes</div>;
  return (
    <table>
      <thead>
        <tr>
          <th>id</th>
          <th>label</th>
          <th className="num">size</th>
          <th className="num">mean fwd ret</th>
        </tr>
      </thead>
      <tbody>
        {regimes.map((r) => (
          <tr
            key={r.regime_id}
            style={
              r.regime_id === currentRegime
                ? { background: "rgba(78,121,167,0.18)" }
                : undefined
            }
          >
            <td>
              <span className="swatch" style={{ background: r.color }} />
              {r.regime_id}
            </td>
            <td>{r.label || "--"}</td>
            <td className="num">{r.size}</td>
            <td
              className={
                "num " + (r.mean_fwd_return >= 0 ? "pos" : "neg")
              }
            >
              {(r.mean_fwd_return * 100).toFixed(3)}%
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
