import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { StepResult, Regime, Metrics } from "./types/contracts";
import {
  loadRun,
  loadMetrics,
  computeMetrics,
  deriveRegimes,
  listRuns,
  listDatasets,
  type DatasetInfo,
} from "./lib/api";
import { startLiveRun, type RunParams } from "./lib/stream";
import Header from "./components/Header";
import SimControl, { SPEEDS } from "./components/SimControl";
import ManifoldMap from "./components/ManifoldMap";
import EquityChart from "./components/EquityChart";
import SignalsChart from "./components/SignalsChart";
import RegimeLegend from "./components/RegimeLegend";
import FeatureHeatmap from "./components/FeatureHeatmap";
import OrderBlotter from "./components/OrderBlotter";

const LIVE_CAP = 50000; // hard cap on retained live points

export default function App() {
  const [steps, setSteps] = useState<StepResult[]>([]);
  const [regimes, setRegimes] = useState<Regime[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [source, setSource] = useState<"api" | "sample" | null>(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);
  const [status, setStatus] = useState("idle");
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [simDataset, setSimDataset] = useState("vix");
  const [simWindow, setSimWindow] = useState(1000);
  const [simStartFrac, setSimStartFrac] = useState(0.5);
  const [simSpeed, setSimSpeed] = useState("fast");

  const connRef = useRef<{ close: () => void } | null>(null);
  // batch live steps to avoid a setState per WS message
  const pendingRef = useRef<StepResult[]>([]);
  const rafRef = useRef<number | null>(null);

  const loadById = useCallback(async (id?: string) => {
    const res = await loadRun(id || undefined);
    setSteps(res.run.steps);
    setRegimes(res.regimes);
    setSource(res.source);
    const m = await loadMetrics(res.run.run_id, res.run.steps);
    setMetrics(m);
    setLoading(false);
  }, []);

  // initial load: discover runs + datasets, default to the real VIX backtest
  useEffect(() => {
    let alive = true;
    (async () => {
      const [ids, ds] = await Promise.all([listRuns(), listDatasets()]);
      if (!alive) return;
      setRuns(ids);
      setDatasets(ds);
      if (ds.length && !ds.some((d) => d.name === "vix")) setSimDataset(ds[0].name);
      // Default to the first advertised run (manifest orders the compact,
      // deepest-history dataset first for a fast mobile first paint).
      const def = ids[0] ?? "";
      setSelectedRun(def);
      await loadById(def);
    })();
    return () => {
      alive = false;
    };
  }, [loadById]);

  const onSelectRun = useCallback(
    (id: string) => {
      if (live) return;
      setSelectedRun(id);
      void loadById(id);
    },
    [live, loadById]
  );

  const flush = useCallback(() => {
    rafRef.current = null;
    if (!pendingRef.current.length) return;
    const batch = pendingRef.current;
    pendingRef.current = [];
    setSteps((prev) => {
      const next = prev.concat(batch);
      return next.length > LIVE_CAP ? next.slice(next.length - LIVE_CAP) : next;
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafRef.current == null)
      rafRef.current = requestAnimationFrame(flush);
  }, [flush]);

  const stopLive = useCallback(() => {
    connRef.current?.close();
    connRef.current = null;
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    pendingRef.current = [];
    setLive(false);
    setStatus("idle");
  }, []);

  const startSim = useCallback(
    async (params: RunParams) => {
      setLive(true);
      setSteps([]);
      setMetrics(null);
      pendingRef.current = [];
      setStatus("starting...");
      connRef.current = await startLiveRun(
        {
          onStatus: (s) => setStatus(s),
          onStart: () => setSource("api"),
          onRegimes: (r) => setRegimes(r),
          onStep: (step) => {
            pendingRef.current.push(step);
            scheduleFlush();
          },
          onEnd: () => setStatus("run ended"),
        },
        params
      );
    },
    [scheduleFlush]
  );

  // Derive the run request + a step estimate from the simulation controls.
  const selectedInfo = useMemo(
    () => datasets.find((d) => d.name === simDataset) ?? null,
    [datasets, simDataset]
  );
  const isSynth = simDataset === "synthetic";
  const totalBars = isSynth ? simWindow : selectedInfo?.n_bars ?? simWindow;
  const maxStart = Math.max(0, totalBars - simWindow);
  const startIdx = isSynth ? 0 : Math.min(maxStart, Math.round(simStartFrac * maxStart));
  const simParams: RunParams = useMemo(
    () =>
      isSynth
        ? {
            dataset: "synthetic",
            mode: "paper",
            speed: SPEEDS[simSpeed],
            n_bars: simWindow,
            persist: false,
          }
        : {
            dataset: simDataset,
            mode: "paper",
            speed: SPEEDS[simSpeed],
            start: startIdx,
            end: startIdx + simWindow,
            persist: false,
          },
    [isSynth, simDataset, simWindow, simSpeed, startIdx]
  );
  const estTrain = Math.min(isSynth ? 400 : 500, Math.max(60, Math.floor(simWindow / 3)));
  const simExpected = Math.max(1, simWindow - estTrain);

  const onToggleLive = useCallback(() => {
    if (live) stopLive();
    else void startSim(simParams);
  }, [live, startSim, stopLive, simParams]);

  useEffect(() => () => connRef.current?.close(), []);

  // recompute metrics locally while live (server metrics are post-run)
  useEffect(() => {
    if (live && steps.length) setMetrics(computeMetrics(steps));
  }, [live, steps]);

  // ensure regimes exist even if server hasn't emitted regime_fit yet
  const effRegimes = useMemo(
    () => (regimes.length ? regimes : deriveRegimes(steps)),
    [regimes, steps]
  );

  const last = steps.length ? steps[steps.length - 1] : null;
  const currentRegime = last?.manifold.regime_id ?? -1;
  const netExposure = last?.portfolio.net_exposure ?? 0;

  if (loading) return <div className="loading">LOADING MANNOFOLD…</div>;

  return (
    <div className="app">
      <Header
        metrics={metrics}
        currentRegime={currentRegime}
        netExposure={netExposure}
        regimes={effRegimes}
        source={source}
        live={live}
        status={status}
        onToggleLive={onToggleLive}
        runs={runs}
        selectedRun={selectedRun}
        onSelectRun={onSelectRun}
      />
      <SimControl
        datasets={datasets}
        dataset={simDataset}
        setDataset={setSimDataset}
        window={simWindow}
        setWindow={setSimWindow}
        startFrac={simStartFrac}
        setStartFrac={setSimStartFrac}
        speed={simSpeed}
        setSpeed={setSimSpeed}
        startIdx={startIdx}
        totalBars={totalBars}
        info={selectedInfo}
        live={live}
        stepCount={steps.length}
        expected={simExpected}
        onRun={() => void startSim(simParams)}
        onStop={stopLive}
      />
      <div className="grid">
        <div className="panel manifold-panel">
          <div className="title">
            <span>manifold map</span>
            <span>{steps.length.toLocaleString()} states</span>
          </div>
          <div className="body">
            <ManifoldMap steps={steps} regimes={effRegimes} />
          </div>
        </div>

        <div className="panel">
          <div className="title">equity / drawdown</div>
          <div className="body">
            <EquityChart steps={steps} />
          </div>
        </div>

        <div className="panel">
          <div className="title">regimes</div>
          <div className="body">
            <RegimeLegend regimes={effRegimes} currentRegime={currentRegime} />
          </div>
        </div>

        <div className="panel">
          <div className="title">signals</div>
          <div className="body">
            <SignalsChart steps={steps} />
          </div>
        </div>

        <div className="panel">
          <div className="title">features (latest)</div>
          <div className="body">
            <FeatureHeatmap step={last} />
          </div>
        </div>

        <div className="panel">
          <div className="title">order blotter</div>
          <div className="body">
            <OrderBlotter steps={steps} />
          </div>
        </div>
      </div>
    </div>
  );
}
