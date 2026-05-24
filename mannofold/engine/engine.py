"""The Mannofold engine: one online step, two clocks.

Phases per run:
  1. accumulate ``train_size`` bars (no trading);
  2. walk-forward fit (scaler → φ → regimes → forward-return model) on TRAIN only;
  3. online: for every subsequent bar, run the single inference step and trade;
     periodically refit on an expanding/rolling window of PAST bars.

Because the online step never reads a future bar and refits see only past data,
the backtest result is reproducible by the paper path bar-for-bar.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mannofold.contracts.events import StreamEvent
from mannofold.contracts.interfaces import DataFeed, RiskSizer, StateStore, Strategy
from mannofold.contracts.models import (
    Bar,
    Fill,
    Mode,
    PortfolioState,
    Regime,
    Side,
    StepResult,
)
from mannofold.features import indicators as ind
from mannofold.features.pipeline import RollingFeaturePipeline
from mannofold.manifold.model import ManifoldModelImpl
from mannofold.signals.risk import VolTargetRiskSizer
from mannofold.signals.strategy import ManifoldStrategy

EventHook = Callable[[StreamEvent], None]


@dataclass
class EngineConfig:
    train_size: int = 800
    refit_every: int = 400
    max_train: int = 1500
    fwd_horizon: int = 10
    n_regimes: int = 4
    n_neighbors: int = 25
    embedder: str = "pca"
    n_components: int = 3
    velocity_lookback: int = 5
    vol_window: int = 20
    target_vol: float = 0.01
    max_leverage: float = 1.0
    rebalance_band: float = 0.02
    commission_bps: float = 1.0
    initial_cash: float = 100_000.0


@dataclass
class RunResult:
    run_id: str
    mode: Mode
    results: list[StepResult]
    regimes: list[Regime] = field(default_factory=list)


def build_training(
    bars: Sequence[Bar], pipe: RollingFeaturePipeline, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + aligned realized forward returns (NaN where unrealized)."""
    pipe.fit(bars)
    closes = [b.close for b in bars]
    rows: list[list[float]] = []
    fwd: list[float] = []
    for i in range(pipe.warmup - 1, len(bars)):
        fv = pipe.transform(bars[i - pipe.warmup + 1 : i + 1])
        rows.append(fv.values)
        if i + horizon < len(bars):
            fwd.append(math.log(closes[i + horizon] / max(closes[i], 1e-9)))
        else:
            fwd.append(float("nan"))
    return np.asarray(rows, dtype=float), np.asarray(fwd, dtype=float)


class Engine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        strategy: Strategy | None = None,
        risk: RiskSizer | None = None,
        store: StateStore | None = None,
        on_event: EventHook | None = None,
        run_id: str = "run",
    ):
        self.cfg = config or EngineConfig()
        self.strategy = strategy or ManifoldStrategy()
        self.risk = risk or VolTargetRiskSizer(
            target_vol=self.cfg.target_vol,
            max_leverage=self.cfg.max_leverage,
            rebalance_band=self.cfg.rebalance_band,
            commission_bps=self.cfg.commission_bps,
        )
        self.store = store
        self.on_event = on_event
        self.run_id = run_id
        self._pipe: RollingFeaturePipeline | None = None
        self._model: ManifoldModelImpl | None = None

    def _fit(self, train_bars: Sequence[Bar]) -> None:
        pipe = RollingFeaturePipeline()
        X, fwd = build_training(train_bars, pipe, self.cfg.fwd_horizon)
        model = ManifoldModelImpl(
            embedder=self.cfg.embedder,
            n_components=self.cfg.n_components,
            n_regimes=self.cfg.n_regimes,
            n_neighbors=self.cfg.n_neighbors,
        )
        model.fit(X, fwd)
        self._pipe, self._model = pipe, model

    def _emit(self, event: StreamEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def run(self, feed: DataFeed) -> RunResult:
        cfg = self.cfg
        buf: list[Bar] = []
        results: list[StepResult] = []
        emb_hist: deque[list[float]] = deque(maxlen=cfg.velocity_lookback + 1)

        cash = cfg.initial_cash
        positions: dict[str, float] = {}
        prev_equity = cfg.initial_cash
        peak_equity = cfg.initial_cash
        fitted = False
        next_refit = 0
        seq = 0

        self._emit(
            StreamEvent(type="run_start", run_id=self.run_id, seq=0, payload={"mode": feed.mode.value})  # type: ignore[arg-type]
        )

        for bar in feed.stream():
            buf.append(bar)
            price = bar.close

            if not fitted:
                if len(buf) >= cfg.train_size:
                    self._fit(buf[-cfg.max_train :])
                    fitted = True
                    next_refit = len(buf) + cfg.refit_every
                    self._emit(StreamEvent.regimes(self.run_id, seq, self._model.regimes))
                continue

            if len(buf) >= next_refit:
                self._fit(buf[-cfg.max_train :])
                next_refit += cfg.refit_every
                self._emit(StreamEvent.regimes(self.run_id, seq, self._model.regimes))

            assert self._pipe is not None and self._model is not None
            window = buf[-self._pipe.warmup :]
            fv = self._pipe.transform(window)
            ms = self._model.transform_online(np.asarray(fv.values))

            emb_hist.append(ms.embedding)
            if len(emb_hist) > cfg.velocity_lookback:
                vel = [c - o for c, o in zip(ms.embedding, emb_hist[0], strict=False)]
            else:
                vel = [0.0] * len(ms.embedding)
            ms = ms.model_copy(update={"ts": bar.ts, "symbol": bar.symbol, "velocity": vel})

            signals = self.strategy.signals(ms)
            target = self.strategy.target(signals)

            closes = np.array([b.close for b in buf[-(cfg.vol_window + 1) :]], dtype=float)
            volatility = ind.realized_vol(closes, cfg.vol_window)

            pre = PortfolioState(
                ts=bar.ts,
                cash=cash,
                equity=cash + positions.get(bar.symbol, 0.0) * price,
                positions=dict(positions),
            )
            order = self.risk.size(target, pre, price, ms.anomaly_score, volatility)

            fill = None
            if order is not None:
                cost = order.qty * price
                commission = cost * cfg.commission_bps / 1e4
                if order.side == Side.BUY:
                    cash -= cost + commission
                    positions[bar.symbol] = positions.get(bar.symbol, 0.0) + order.qty
                else:
                    cash += cost - commission
                    positions[bar.symbol] = positions.get(bar.symbol, 0.0) - order.qty
                fill = Fill(
                    ts=bar.ts,
                    symbol=bar.symbol,
                    side=order.side,
                    qty=order.qty,
                    price=price,
                    commission=commission,
                )

            pos_qty = positions.get(bar.symbol, 0.0)
            pos_value = pos_qty * price
            equity = cash + pos_value
            peak_equity = max(peak_equity, equity)
            portfolio = PortfolioState(
                ts=bar.ts,
                cash=cash,
                equity=equity,
                gross_exposure=abs(pos_value) / max(equity, 1e-9),
                net_exposure=pos_value / max(equity, 1e-9),
                positions=dict(positions),
                returns=equity / max(prev_equity, 1e-9) - 1.0,
                drawdown=equity / max(peak_equity, 1e-9) - 1.0,
            )
            prev_equity = equity

            sr = StepResult(
                seq=seq,
                mode=feed.mode,
                bar=bar,
                features=fv,
                manifold=ms,
                signals=signals,
                target=target,
                order=order,
                fill=fill,
                portfolio=portfolio,
            )
            results.append(sr)
            self._emit(StreamEvent.step(self.run_id, sr))
            seq += 1

        regimes = self._model.regimes if self._model else []
        if self.store is not None:
            self.store.append_bars(buf)
            self.store.write_run(self.run_id, results)
            self.store.write_regimes(self.run_id, regimes)

        self._emit(
            StreamEvent(type="run_end", run_id=self.run_id, seq=seq, payload={"n_steps": seq})  # type: ignore[arg-type]
        )
        return RunResult(run_id=self.run_id, mode=feed.mode, results=results, regimes=regimes)
