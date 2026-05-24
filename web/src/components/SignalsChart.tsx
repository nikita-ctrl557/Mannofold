import { useMemo } from "react";
import type uPlot from "uplot";
import type { StepResult } from "../types/contracts";
import UPlotChart from "./UPlotChart";
import { baseOpts, timeAxis, numAxis } from "./EquityChart";

interface Props {
  steps: StepResult[];
}

// expected_return, anomaly, target_weight over time (shared time axis,
// two value scales: signal magnitudes vs target weight in [-1,1]).
export default function SignalsChart({ steps }: Props) {
  const data = useMemo<uPlot.AlignedData>(() => {
    const t = new Array<number>(steps.length);
    const er = new Array<number>(steps.length);
    const an = new Array<number>(steps.length);
    const tw = new Array<number>(steps.length);
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      t[i] = Date.parse(s.signals.ts) / 1000;
      er[i] = s.signals.expected_return * 100;
      an[i] = s.signals.anomaly;
      tw[i] = s.target.target_weight;
    }
    return [t, er, an, tw];
  }, [steps]);

  const options = useMemo<Omit<uPlot.Options, "width" | "height">>(
    () => ({
      ...baseOpts(),
      series: [
        {},
        {
          label: "E[ret] %",
          stroke: "#59a14f",
          width: 1,
          scale: "sig",
          value: (_u, v) => (v == null ? "--" : v.toFixed(3)),
        },
        {
          label: "anomaly",
          stroke: "#edc948",
          width: 1,
          scale: "sig",
          value: (_u, v) => (v == null ? "--" : v.toFixed(3)),
        },
        {
          label: "tgt wt",
          stroke: "#b07aa1",
          width: 1,
          scale: "wt",
          value: (_u, v) => (v == null ? "--" : v.toFixed(3)),
        },
      ],
      axes: [
        timeAxis(),
        numAxis("sig", "#6b7686"),
        { ...numAxis("wt", "#b07aa1"), side: 1 },
      ],
      scales: { x: { time: true }, sig: { auto: true }, wt: { auto: true } },
    }),
    []
  );

  if (!steps.length) return <div className="empty">no data</div>;
  return <UPlotChart data={data} options={options} />;
}
