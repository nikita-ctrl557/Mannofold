// Hand-mirrored from contracts.schema.json (Python pydantic source of truth).

export type Side = "buy" | "sell" | "flat";
export type Mode = "backtest" | "paper";
export type EventType =
  | "run_start"
  | "step"
  | "regime_fit"
  | "portfolio"
  | "run_end";

export interface Bar {
  ts: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FeatureVector {
  ts: string;
  symbol: string;
  values: number[];
  names?: string[];
}

export interface ManifoldState {
  ts: string;
  symbol: string;
  embedding: number[];
  regime_id: number;
  regime_prob: number;
  density: number;
  anomaly_score: number;
  fwd_return_mean: number;
  fwd_return_std: number;
  velocity?: number[];
}

export interface SignalSet {
  ts: string;
  symbol: string;
  momentum: number;
  expected_return: number;
  anomaly: number;
  regime_id: number;
  confidence: number;
}

export interface TargetPosition {
  ts: string;
  symbol: string;
  target_weight: number;
}

export interface Order {
  ts: string;
  symbol: string;
  side: Side;
  qty: number;
  target_weight: number;
  reason: string;
}

export interface Fill {
  ts: string;
  symbol: string;
  side: Side;
  qty: number;
  price: number;
  commission: number;
}

export interface PortfolioState {
  ts: string;
  cash: number;
  equity: number;
  gross_exposure: number;
  net_exposure: number;
  positions: Record<string, number>;
  returns: number;
  drawdown: number;
}

export interface StepResult {
  seq: number;
  mode: Mode;
  bar: Bar;
  features: FeatureVector;
  manifold: ManifoldState;
  signals: SignalSet;
  target: TargetPosition;
  order: Order | null;
  fill: Fill | null;
  portfolio: PortfolioState;
}

export interface Regime {
  regime_id: number;
  label: string;
  color: string;
  size: number;
  mean_fwd_return: number;
}

export interface StreamEvent {
  type: EventType;
  run_id: string;
  seq: number;
  payload: Record<string, unknown>;
}

export interface RunData {
  run_id: string;
  steps: StepResult[];
}

export interface Metrics {
  n_steps: number;
  n_trades: number;
  total_return: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  final_equity: number;
}
