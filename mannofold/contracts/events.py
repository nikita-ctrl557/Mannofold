"""WebSocket event schema + JSON-Schema export for TypeScript codegen.

The frontend consumes exactly these events. Run ``python -m mannofold.contracts.events``
to write ``web/src/types/contracts.schema.json``, the single source for the TS types.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mannofold.contracts.models import (
    PortfolioState,
    Regime,
    StepResult,
)


class EventType(str, Enum):
    RUN_START = "run_start"
    STEP = "step"
    REGIME_FIT = "regime_fit"
    PORTFOLIO = "portfolio"
    RUN_END = "run_end"


class StreamEvent(BaseModel):
    """Envelope for everything pushed over the WebSocket.

    ``seq`` is monotonic per run so the client can detect drops (the server may
    drop-oldest under backpressure). ``payload`` shape depends on ``type``.
    """

    type: EventType
    run_id: str
    seq: int
    payload: dict[str, Any]

    @classmethod
    def step(cls, run_id: str, result: StepResult) -> StreamEvent:
        return cls(type=EventType.STEP, run_id=run_id, seq=result.seq, payload=result.model_dump(mode="json"))

    @classmethod
    def regimes(cls, run_id: str, seq: int, regimes: list[Regime]) -> StreamEvent:
        return cls(
            type=EventType.REGIME_FIT,
            run_id=run_id,
            seq=seq,
            payload={"regimes": [r.model_dump(mode="json") for r in regimes]},
        )

    @classmethod
    def portfolio(cls, run_id: str, seq: int, state: PortfolioState) -> StreamEvent:
        return cls(type=EventType.PORTFOLIO, run_id=run_id, seq=seq, payload=state.model_dump(mode="json"))


_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "StreamEvent": StreamEvent,
    "StepResult": StepResult,
    "Regime": Regime,
    "PortfolioState": PortfolioState,
}


def export_json_schema() -> dict[str, Any]:
    return {name: model.model_json_schema() for name, model in _SCHEMA_MODELS.items()}


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "web" / "src" / "types" / "contracts.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export_json_schema(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
