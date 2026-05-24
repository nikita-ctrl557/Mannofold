import type { StepResult } from "../types/contracts";

interface Props {
  step: StepResult | null;
}

// Diverging color bar per feature of the latest StepResult.features.
// Values are roughly standardized; clamp to [-3,3] for color mapping.
export default function FeatureHeatmap({ step }: Props) {
  if (!step) return <div className="empty">no data</div>;
  const { values, names } = step.features;
  return (
    <div className="heat">
      {values.map((v, i) => {
        const name = names?.[i] ?? `f${i}`;
        const clamped = Math.max(-3, Math.min(3, v));
        const t = clamped / 3; // -1..1
        const color =
          t >= 0
            ? `rgba(89,161,79,${0.2 + 0.7 * t})`
            : `rgba(225,87,89,${0.2 + 0.7 * -t})`;
        const widthPct = 50 + t * 50; // center-anchored fill
        const left = t >= 0 ? 50 : widthPct;
        return (
          <div className="cell" key={i}>
            <div className="name" title={name}>
              {name}
            </div>
            <div className="bar-wrap">
              <div
                className="bar"
                style={{
                  left: `${left}%`,
                  width: `${Math.abs(t) * 50}%`,
                  background: color,
                }}
              />
            </div>
            <div className="v">{v.toFixed(3)}</div>
          </div>
        );
      })}
    </div>
  );
}
