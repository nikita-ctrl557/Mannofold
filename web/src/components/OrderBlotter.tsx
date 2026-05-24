import { useMemo } from "react";
import type { StepResult } from "../types/contracts";

interface Props {
  steps: StepResult[];
  limit?: number;
}

// Most-recent fills, newest first.
export default function OrderBlotter({ steps, limit = 200 }: Props) {
  const fills = useMemo(() => {
    const out: StepResult["fill"][] = [];
    for (let i = steps.length - 1; i >= 0 && out.length < limit; i--) {
      if (steps[i].fill) out.push(steps[i].fill);
    }
    return out;
  }, [steps, limit]);

  if (!fills.length) return <div className="empty">no fills</div>;
  return (
    <table>
      <thead>
        <tr>
          <th>ts</th>
          <th>side</th>
          <th className="num">qty</th>
          <th className="num">price</th>
        </tr>
      </thead>
      <tbody>
        {fills.map((f, i) => (
          <tr key={i}>
            <td>{fmtTs(f!.ts)}</td>
            <td className={f!.side === "buy" ? "side-buy" : "side-sell"}>
              {f!.side}
            </td>
            <td className="num">{f!.qty.toFixed(2)}</td>
            <td className="num">{f!.price.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmtTs(ts: string): string {
  return ts.replace("T", " ").replace("Z", "").slice(5, 16);
}
