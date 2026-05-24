"""Local DuckDB + Parquet state store.

Write model (validated by the architecture review): never stream row-by-row into
one Parquet file (metadata lives at the file tail and DuckDB is single-writer).
Instead write immutable micro-batch Parquet parts into a partitioned folder and
query them through a DuckDB glob view. Run artifacts are batch-written at run end.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from mannofold.contracts.models import Bar, Regime, StepResult

DEFAULT_DATA_DIR = Path("data")


class LocalStateStore:
    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        (self.data_dir / "bars").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs").mkdir(parents=True, exist_ok=True)

    def append_bars(self, bars: Sequence[Bar]) -> None:
        if not bars:
            return
        df = pd.DataFrame(b.model_dump() for b in bars)
        for symbol, sub in df.groupby("symbol"):
            part_dir = self.data_dir / "bars" / f"symbol={symbol}"
            part_dir.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(part_dir / f"part-{uuid.uuid4().hex}.parquet", index=False)

    def write_run(self, run_id: str, results: Sequence[StepResult]) -> None:
        run_dir = self.data_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = [self._flatten(r) for r in results]
        if rows:
            pd.DataFrame(rows).to_parquet(run_dir / "steps.parquet", index=False)
        # Convenience snapshot for the frontend (full fidelity, JSON).
        (run_dir / "run.json").write_text(
            json.dumps({"run_id": run_id, "steps": [r.model_dump(mode="json") for r in results]})
        )

    def write_regimes(self, run_id: str, regimes: Sequence[Regime]) -> None:
        run_dir = self.data_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "regimes.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in regimes])
        )

    def query(self, sql: str) -> list[dict]:
        con = duckdb.connect()
        try:
            con.execute(
                f"CREATE VIEW bars AS SELECT * FROM read_parquet('{self.data_dir}/bars/*/*.parquet', union_by_name=true)"
            )
        except duckdb.Error:
            pass
        return con.execute(sql).df().to_dict(orient="records")

    @staticmethod
    def _flatten(r: StepResult) -> dict:
        emb = (r.manifold.embedding + [0.0, 0.0, 0.0])[:3]
        return {
            "seq": r.seq,
            "ts": r.bar.ts,
            "symbol": r.bar.symbol,
            "close": r.bar.close,
            "emb_x": emb[0],
            "emb_y": emb[1],
            "emb_z": emb[2],
            "regime_id": r.manifold.regime_id,
            "anomaly": r.manifold.anomaly_score,
            "expected_return": r.signals.expected_return,
            "target_weight": r.target.target_weight,
            "equity": r.portfolio.equity,
            "drawdown": r.portfolio.drawdown,
            "net_exposure": r.portfolio.net_exposure,
        }
