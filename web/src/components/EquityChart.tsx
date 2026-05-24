import { useMemo } from "react";
import type uPlot from "uplot";
import type { StepResult } from "../types/contracts";
import UPlotChart from "./UPlotChart";

interface Props {
  steps: StepResult[];
}

// Equity curve (left axis) + drawdown (right axis, filled negative band).
export default function EquityChart({ steps }: Props) {
  const data = useMemo<uPlot.AlignedData>(() => {
    const t = new Array<number>(steps.length);
    const eq = new Array<number>(steps.length);
    const dd = new Array<number>(steps.length);
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      t[i] = Date.parse(s.portfolio.ts) / 1000;
      eq[i] = s.portfolio.equity;
      dd[i] = s.portfolio.drawdown * 100;
    }
    return [t, eq, dd];
  }, [steps]);

  const options = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      ...baseOpts(),
      series: [
        {},
        {
          label: "equity",
          stroke: "#4e79a7",
          width: 1.5,
          scale: "eq",
          value: (_u, v) => (v == null ? "--" : v.toFixed(0)),
        },
        {
          label: "drawdown %",
          stroke: "#e15759",
          fill: "rgba(225,87,89,0.12)",
          width: 1,
          scale: "dd",
          value: (_u, v) => (v == null ? "--" : v.toFixed(2) + "%"),
        },
      ],
      axes: [
        timeAxis(),
        numAxis("eq", "#4e79a7"),
        { ...numAxis("dd", "#e15759"), side: 1 },
      ],
      scales: {
        x: { time: true },
        eq: { auto: true },
        dd: { auto: true },
      },
    }),
    []
  );

  if (!steps.length) return <div className="empty">no data</div>;
  return <UPlotChart data={data} options={options} />;
}

export function baseOpts(): Omit<uPlot.Options, "width" | "height" | "series"> {
  return {
    cursor: { drag: { x: true, y: false } },
    legend: { live: true },
    padding: [6, 8, 2, 4],
  } as Omit<uPlot.Options, "width" | "height" | "series">;
}

export function timeAxis(): uPlot.Axis {
  return {
    stroke: "#6b7686",
    grid: { stroke: "rgba(35,42,54,0.6)", width: 1 },
    ticks: { stroke: "rgba(35,42,54,0.6)", width: 1 },
    font: "10px monospace",
  };
}

export function numAxis(scale: string, stroke: string): uPlot.Axis {
  return {
    scale,
    stroke,
    grid: { stroke: "rgba(35,42,54,0.4)", width: 1 },
    ticks: { stroke: "rgba(35,42,54,0.4)", width: 1 },
    font: "10px monospace",
    size: 46,
  };
}
