import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

interface Props {
  data: uPlot.AlignedData;
  options: Omit<uPlot.Options, "width" | "height">;
}

// Thin React wrapper around uPlot that auto-sizes to its container via
// ResizeObserver. Re-creates the plot on option changes, setData on data.
export default function UPlotChart({ data, options }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const optsRef = useRef(options);
  optsRef.current = options;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const make = () => {
      const w = host.clientWidth || 300;
      const h = host.clientHeight || 150;
      if (plotRef.current) plotRef.current.destroy();
      plotRef.current = new uPlot(
        { ...optsRef.current, width: w, height: h },
        data,
        host
      );
    };
    make();

    const ro = new ResizeObserver(() => {
      const p = plotRef.current;
      if (!p || !host) return;
      p.setSize({
        width: host.clientWidth || 300,
        height: host.clientHeight || 150,
      });
    });
    ro.observe(host);

    return () => {
      ro.disconnect();
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // re-create only when the series structure (options) changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options]);

  useEffect(() => {
    plotRef.current?.setData(data);
  }, [data]);

  return <div className="chart-host" ref={hostRef} />;
}
