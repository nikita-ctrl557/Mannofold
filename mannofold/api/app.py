"""FastAPI app: REST control plane + WebSocket event stream for the engine.

Architecture (the load-bearing part)
-------------------------------------
The engine's ``on_event`` callback is **synchronous** and is invoked from a
worker thread (we submit ``Engine.run`` to ``loop.run_in_executor``). The
WebSocket fan-out is **async**. We bridge the two without ever letting a slow
client stall the engine:

* Each run gets a :class:`Hub` (a pub/sub registry of subscribers).
* Each WS client owns a *bounded* ``asyncio.Queue`` (``maxsize`` ~1000). On
  overflow we drop the **oldest** event for that client only (the ``seq`` field
  lets the client detect the gap).
* A per-connection *writer task* drains that client's queue to the socket.
* The engine thread publishes via ``loop.call_soon_threadsafe`` — the only
  thread-safe way to touch asyncio primitives from another thread. The
  publish is O(n_subscribers) enqueue-or-drop and never blocks on a socket,
  so the engine loop runs at full speed regardless of client speed.

Late subscribers are supported: each hub keeps a bounded replay buffer of the
events seen so far, so a client that connects shortly after the run starts is
seeded with the history (from ``run_start``) before live events resume. When a
run has already finished, the hub is marked closed; new subscribers still get
the buffered history followed by the close sentinel, then the socket closes.
Runs longer than the buffer only replay their most recent window.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mannofold.contracts.events import StreamEvent
from mannofold.contracts.models import StepResult
from mannofold.engine.engine import Engine, EngineConfig
from mannofold.engine.metrics import compute_metrics
from mannofold.feed.github_csv import DATASETS, load_bars
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.live_replay import LiveReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore

CLIENT_QUEUE_MAXSIZE = 1000
# Bounded per-run history so a subscriber that connects shortly after the run
# starts still receives the early events (notably ``run_start``). On runs longer
# than this, only the most recent ``REPLAY_BUFFER_MAXLEN`` events are replayed.
REPLAY_BUFFER_MAXLEN = 100_000


class Hub:
    """Per-run pub/sub. One bounded queue per subscriber; drop-oldest on overflow.

    All mutation happens on the event loop thread, so the only cross-thread entry
    point is :meth:`publish_threadsafe`, which schedules ``_publish`` onto the loop.

    A bounded replay buffer holds the events seen so far; a new subscriber is
    seeded with that history (oldest first) at subscribe time, so connecting a
    few milliseconds after the engine starts still yields the full stream from
    ``run_start``. The buffer never blocks the engine: publishing is still an
    O(n_subscribers) enqueue-or-drop on the loop thread.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self._history: deque[dict[str, Any] | None] = deque(maxlen=REPLAY_BUFFER_MAXLEN)
        self.closed = False

    def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=CLIENT_QUEUE_MAXSIZE)
        # Seed with buffered history first (drop-oldest if it exceeds the queue),
        # then register for live events. Both run on the loop thread, so no event
        # can interleave between replay and registration.
        for item in self._history:
            self._offer(q, item)
        self._subscribers.add(q)
        if self.closed:
            # Already-finished run: ensure the writer sees a sentinel and exits
            # instead of blocking forever on ``queue.get()``.
            self._offer(q, None)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._subscribers.discard(q)

    @staticmethod
    def _offer(q: asyncio.Queue[dict[str, Any] | None], item: dict[str, Any] | None) -> None:
        """Enqueue, dropping this client's oldest event on overflow (never blocks)."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover - defensive
            pass

    def _publish(self, item: dict[str, Any] | None) -> None:
        """Runs on the loop thread. Buffer + enqueue to every client."""
        self._history.append(item)
        for q in self._subscribers:
            self._offer(q, item)

    def publish_threadsafe(self, event: StreamEvent) -> None:
        """Called from the engine worker thread. Hands the event to the loop."""
        payload = event.model_dump(mode="json")
        self._loop.call_soon_threadsafe(self._publish, payload)

    def close_threadsafe(self) -> None:
        """Mark the hub closed and signal subscribers (sentinel ``None``)."""

        def _close() -> None:
            self.closed = True
            self._publish(None)

        self._loop.call_soon_threadsafe(_close)


# Module-level registry: run_id -> Hub.
_HUBS: dict[str, Hub] = {}


ALLOWED_DATASETS = ("synthetic", *DATASETS.keys())


class StartRunBody(BaseModel):
    dataset: str = "synthetic"  # "synthetic" | "vix" | "aapl" | "sp500"
    n_bars: int = 1500  # synthetic only
    seed: int = 7  # synthetic only
    mode: str = "backtest"  # "backtest" | "paper" (paper honours `speed`)
    speed: float = 0.0  # seconds between bars; 0 = ultraspeed
    start: int | None = None  # inclusive bar index — pick a slice of history
    end: int | None = None  # exclusive bar index
    persist: bool = True  # sim/replay runs set false to avoid cluttering /api/runs


def _data_dir() -> Path:
    """Resolve the data dir at call time so tests can override the cwd."""
    return Path("data")


def _build_run(body: StartRunBody):
    """Build the (feed, config) for a run from its dataset + history segment.

    The same engine serves every dataset; only the feed and walk-forward windows
    differ. Slicing ``[start:end]`` is how the UI blind-backtests an arbitrary
    slice of history (e.g. the 2008 or 2020 regime).
    """
    if body.dataset == "synthetic":
        bars, _ = generate_bars(SyntheticConfig(n_bars=body.n_bars, seed=body.seed))
        base_train, refit_every, max_train = min(400, max(60, body.n_bars // 3)), 400, 1500
    else:
        bars = load_bars(body.dataset)
        base_train, refit_every, max_train = 500, 250, 1000

    if body.start is not None or body.end is not None:
        s = max(0, body.start or 0)
        e = body.end if body.end is not None else len(bars)
        bars = bars[s:e]
    if len(bars) < 120:
        raise ValueError("history segment too short (need >= 120 bars)")

    train_size = min(base_train, max(60, len(bars) // 3))
    cfg = EngineConfig(train_size=train_size, refit_every=refit_every, max_train=max_train)
    feed: HistoricalReplayFeed | LiveReplayFeed = (
        LiveReplayFeed(bars, speed=body.speed)
        if body.mode == "paper"
        else HistoricalReplayFeed(bars)
    )
    return feed, cfg


def _run_engine(
    run_id: str,
    body: StartRunBody,
    data_dir: Path,
    publish: Callable[[StreamEvent], None],
    on_done: Callable[[], None],
) -> None:
    """Synchronous engine driver — runs inside the executor thread."""
    try:
        feed, cfg = _build_run(body)
        store = LocalStateStore(data_dir) if body.persist else None
        engine = Engine(config=cfg, store=store, on_event=publish, run_id=run_id)
        engine.run(feed)
    finally:
        on_done()


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Mannofold API")

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolve_dir = (lambda: data_dir) if data_dir is not None else _data_dir

    def runs_dir() -> Path:
        return resolve_dir() / "runs"

    @app.get("/api/runs")
    def list_runs() -> dict[str, list[str]]:
        d = runs_dir()
        if not d.exists():
            return {"runs": []}
        return {"runs": sorted(p.name for p in d.iterdir() if p.is_dir())}

    @app.get("/api/datasets")
    def list_datasets() -> dict[str, list[dict[str, Any]]]:
        """Available datasets for the simulation view, with span metadata."""
        out: list[dict[str, Any]] = [
            {
                "name": "synthetic",
                "symbol": "SYNTH",
                "n_bars": None,
                "start": None,
                "end": None,
                "description": "Regime-switching synthetic market (configurable length).",
            }
        ]
        for name, ds in DATASETS.items():
            try:
                bars = load_bars(name)
            except Exception:  # pragma: no cover - network/parse issues
                continue
            out.append(
                {
                    "name": name,
                    "symbol": ds.symbol,
                    "n_bars": len(bars),
                    "start": bars[0].ts.isoformat(),
                    "end": bars[-1].ts.isoformat(),
                    "description": ds.description,
                }
            )
        return {"datasets": out}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        path = runs_dir() / run_id / "run.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return JSONResponse(content=json.loads(path.read_text()))

    @app.get("/api/runs/{run_id}/regimes")
    def get_regimes(run_id: str) -> JSONResponse:
        path = runs_dir() / run_id / "regimes.json"
        if not path.exists():
            return JSONResponse(content=[])
        return JSONResponse(content=json.loads(path.read_text()))

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str) -> dict[str, Any]:
        path = runs_dir() / run_id / "run.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        data = json.loads(path.read_text())
        steps = [StepResult.model_validate(s) for s in data.get("steps", [])]
        return compute_metrics(steps)

    @app.post("/api/runs")
    async def start_run(body: StartRunBody | None = None) -> dict[str, str]:
        body = body or StartRunBody()
        if body.mode not in ("backtest", "paper"):
            raise HTTPException(status_code=422, detail="mode must be 'backtest' or 'paper'")
        if body.dataset not in ALLOWED_DATASETS:
            raise HTTPException(status_code=422, detail=f"dataset must be one of {ALLOWED_DATASETS}")

        run_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()  # capture BEFORE submitting to the executor
        hub = Hub(loop)
        _HUBS[run_id] = hub

        def publish(event: StreamEvent) -> None:
            hub.publish_threadsafe(event)

        def on_done() -> None:
            hub.close_threadsafe()

        # Fire-and-forget on the default executor; returns immediately.
        loop.run_in_executor(
            None, _run_engine, run_id, body, resolve_dir(), publish, on_done
        )
        return {"run_id": run_id}

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket, run_id: str = Query(...)) -> None:
        await websocket.accept()
        hub = _HUBS.get(run_id)
        if hub is None:
            # Unknown or already-evicted run: nothing to stream.
            await websocket.send_json(
                {"type": "error", "run_id": run_id, "seq": -1, "payload": {"detail": "no such run"}}
            )
            await websocket.close()
            return

        queue = hub.subscribe()
        try:
            while True:
                item = await queue.get()
                if item is None:  # run finished -> sentinel
                    break
                await websocket.send_json(item)
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(queue)
        await _safe_close(websocket)

    # Optionally serve the built frontend if present (mounted last so /api + /ws win).
    web_dist = Path("web/dist")
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="static")

    return app


async def _safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:  # pragma: no cover - socket already closed
        pass


app = create_app()
