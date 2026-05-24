import { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, PathLayer } from "@deck.gl/layers";
import { OrthographicView } from "@deck.gl/core";
import type { StepResult, Regime } from "../types/contracts";
import { hexToRgb, regimeColorMap } from "../lib/api";

interface Props {
  steps: StepResult[];
  regimes: Regime[];
}

interface Hover {
  x: number;
  y: number;
  step: StepResult;
}

const TRAIL = 50;

export default function ManifoldMap({ steps, regimes }: Props) {
  const [hover, setHover] = useState<Hover | null>(null);

  const colorMap = useMemo(() => regimeColorMap(regimes), [regimes]);

  // Precompute view bounds from the embedding so the scatter auto-fits.
  const viewState = useMemo(() => {
    if (!steps.length)
      return { target: [0, 0, 0], zoom: 5 } as Record<string, unknown>;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const s of steps) {
      const e = s.manifold.embedding;
      const x = e[0] ?? 0;
      const y = e[1] ?? 0;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const span = Math.max(maxX - minX, maxY - minY, 1e-3);
    const zoom = Math.log2(360 / span); // heuristic fit for ~600px host
    return { target: [cx, cy, 0], zoom };
  }, [steps]);

  const layers = useMemo(() => {
    const n = steps.length;
    const scatter = new ScatterplotLayer<StepResult>({
      id: "manifold-scatter",
      data: steps,
      getPosition: (s) => [s.manifold.embedding[0], s.manifold.embedding[1]],
      getFillColor: (s) => {
        const c = colorMap.get(s.manifold.regime_id) ?? "#666666";
        const [r, g, b] = hexToRgb(c);
        return [r, g, b, 170];
      },
      getRadius: (s) => 1.2 + (s.manifold.density ?? 0) * 1.5,
      radiusUnits: "pixels",
      radiusMinPixels: 1,
      radiusMaxPixels: 6,
      pickable: true,
      onHover: (info) => {
        if (info.object && info.x != null && info.y != null) {
          setHover({ x: info.x, y: info.y, step: info.object as StepResult });
        } else {
          setHover(null);
        }
      },
      updateTriggers: { getFillColor: regimes },
    });

    const trailSteps = steps.slice(Math.max(0, n - TRAIL));
    const path = new PathLayer<{ path: [number, number][] }>({
      id: "manifold-trail",
      data:
        trailSteps.length > 1
          ? [
              {
                path: trailSteps.map(
                  (s) =>
                    [s.manifold.embedding[0], s.manifold.embedding[1]] as [
                      number,
                      number
                    ]
                ),
              },
            ]
          : [],
      getPath: (d) => d.path,
      getColor: [237, 201, 72, 200],
      getWidth: 1.5,
      widthUnits: "pixels",
      widthMinPixels: 1,
    });

    const last = steps[n - 1];
    const head = new ScatterplotLayer<StepResult>({
      id: "manifold-head",
      data: last ? [last] : [],
      getPosition: (s) => [s.manifold.embedding[0], s.manifold.embedding[1]],
      getFillColor: [255, 255, 255, 255],
      getLineColor: [237, 201, 72, 255],
      lineWidthMinPixels: 1.5,
      stroked: true,
      getRadius: 5,
      radiusUnits: "pixels",
    });

    return [scatter, path, head];
  }, [steps, colorMap, regimes]);

  return (
    <div className="deck-host">
      <DeckGL
        views={new OrthographicView({ id: "ortho" })}
        initialViewState={viewState}
        controller={true}
        layers={layers}
      />
      {hover && (
        <div
          className="tooltip"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          <div>
            <span className="k">ts </span>
            {fmtTs(hover.step.manifold.ts)}
          </div>
          <div>
            <span className="k">regime </span>
            {hover.step.manifold.regime_id}{" "}
            {regimes.find((r) => r.regime_id === hover.step.manifold.regime_id)
              ?.label ?? ""}
          </div>
          <div>
            <span className="k">E[ret] </span>
            {(hover.step.signals.expected_return * 100).toFixed(3)}%
          </div>
          <div>
            <span className="k">anomaly </span>
            {hover.step.signals.anomaly.toFixed(3)}
          </div>
          <div>
            <span className="k">density </span>
            {hover.step.manifold.density.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );
}

function fmtTs(ts: string): string {
  return ts.replace("T", " ").replace("Z", "").slice(0, 16);
}
