"""Optional Supabase exporter implementing the :class:`StateStore` Protocol.

Why this is optional and OFF the hot path
------------------------------------------
The Mannofold container is ephemeral. The reference state store
(:class:`~mannofold.persist.store.LocalStateStore`) writes Parquet/JSON to the
local disk and is the only thing the engine needs on its critical path. This
exporter is a *scale-out* convenience: a way to ship a finished run's artifacts
to a shared Supabase Postgres so a long-lived frontend can read them after the
container is gone.

Because of that it is built to NEVER be load-bearing:

* If ``SUPABASE_URL`` / ``SUPABASE_KEY`` are not set it constructs fine and every
  method NO-OPs after emitting a single one-time informational message. Importing
  or instantiating it can never crash for lack of credentials.
* Network failures during export are caught and logged, never raised — a failed
  export must not take down a run.
* It writes via the Supabase REST (PostgREST) API with ``httpx``; no extra deps.

It is therefore safe to wire in as a *secondary* store (e.g. fan-out after
``LocalStateStore`` at run end), but it must never replace the local store on the
engine's inline write path.

Expected table schema (PostgREST upsert targets)
-------------------------------------------------
``bars`` (one row per OHLCV bar)::

    symbol      text
    ts          timestamptz
    open        double precision
    high        double precision
    low         double precision
    close       double precision
    volume      double precision
    primary key (symbol, ts)

``runs`` (one row per run; the JSON snapshot lives here)::

    run_id      text primary key
    n_steps     integer
    created_at  timestamptz default now()

``steps`` (the flattened per-step rows — same columns as
``LocalStateStore._flatten``)::

    run_id           text   references runs(run_id)
    seq              integer
    ts               timestamptz
    symbol           text
    close            double precision
    emb_x, emb_y, emb_z   double precision
    regime_id        integer
    anomaly          double precision
    expected_return  double precision
    target_weight    double precision
    equity           double precision
    drawdown         double precision
    net_exposure     double precision
    primary key (run_id, seq)

``regimes`` (regime metadata per run)::

    run_id          text references runs(run_id)
    regime_id       integer
    label           text
    color           text
    size            integer
    mean_fwd_return double precision
    primary key (run_id, regime_id)
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

from mannofold.contracts.models import Bar, Regime, StepResult
from mannofold.persist.store import LocalStateStore

# Batch size for REST inserts; keeps payloads bounded for long runs.
_BATCH = 500


class SupabaseExporter:
    """A :class:`StateStore` that mirrors run artifacts to Supabase via REST.

    Construct with no arguments to read creds from the environment, or pass them
    explicitly. With no creds the instance is a safe no-op (``self.enabled`` is
    ``False``).
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        self.url = (url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.key = key or os.environ.get("SUPABASE_KEY") or ""
        self.timeout = timeout
        self.enabled = bool(self.url and self.key)
        self._warned = False
        if not self.enabled:
            self._notice(
                "SupabaseExporter: SUPABASE_URL/SUPABASE_KEY not set — "
                "running as a no-op (local store remains the source of truth)."
            )

    # --- StateStore Protocol -------------------------------------------------

    def append_bars(self, bars: Sequence[Bar]) -> None:
        if not self._active(bars):
            return
        rows = [b.model_dump(mode="json") for b in bars]
        self._upsert("bars", rows, on_conflict="symbol,ts")

    def write_run(self, run_id: str, results: Sequence[StepResult]) -> None:
        if not self._active(results):
            return
        self._upsert(
            "runs",
            [{"run_id": run_id, "n_steps": len(results)}],
            on_conflict="run_id",
        )
        rows = [{"run_id": run_id, **LocalStateStore._flatten(r)} for r in results]
        # _flatten emits datetime objects; PostgREST wants JSON-friendly values.
        for row in rows:
            ts = row.get("ts")
            if hasattr(ts, "isoformat"):
                row["ts"] = ts.isoformat()
        self._upsert("steps", rows, on_conflict="run_id,seq")

    def write_regimes(self, run_id: str, regimes: Sequence[Regime]) -> None:
        if not self._active(regimes):
            return
        rows = [{"run_id": run_id, **r.model_dump(mode="json")} for r in regimes]
        self._upsert("regimes", rows, on_conflict="run_id,regime_id")

    def query(self, sql: str) -> list[dict]:
        # PostgREST is not a SQL endpoint; cross-store SQL belongs to the local
        # DuckDB store. Returning [] keeps the Protocol satisfied without lying.
        if not self.enabled:
            return []
        self._notice(
            "SupabaseExporter.query: arbitrary SQL is not supported over REST; "
            "use LocalStateStore.query for SQL access. Returning []."
        )
        return []

    # --- internals -----------------------------------------------------------

    def _active(self, payload: Sequence) -> bool:
        """True only when enabled AND there is something to send."""
        return bool(self.enabled and payload)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            # Upsert semantics + don't return the inserted rows (cheaper).
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }

    def _upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        endpoint = f"{self.url}/rest/v1/{table}"
        params = {"on_conflict": on_conflict}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for start in range(0, len(rows), _BATCH):
                    batch = rows[start : start + _BATCH]
                    resp = client.post(
                        endpoint, params=params, headers=self._headers(), json=batch
                    )
                    resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — export must never crash a run.
            self._notice(f"SupabaseExporter: export to '{table}' failed (ignored): {exc}")

    def _notice(self, msg: str) -> None:
        """Emit an informational message at most once per instance."""
        if self._warned:
            return
        self._warned = True
        print(msg)
