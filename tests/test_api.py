"""API tests: REST shapes + the sync-engine -> async-WS bridge.

We use a tiny engine run (small ``n_bars``) into a tmp data dir, then drive the
app through ``TestClient`` (which runs a real event loop + executor underneath).
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from mannofold.api.app import create_app


def _make_client(tmp_path):
    app = create_app(data_dir=tmp_path)
    return TestClient(app)


def _poll_until_persisted(client, run_id, timeout=60.0):
    """POST returns immediately; the engine persists run.json at the end."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        if resp.status_code == 200:
            return resp.json()
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} never persisted within {timeout}s")


def test_list_runs_empty(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_get_missing_run_is_404(tmp_path):
    client = _make_client(tmp_path)
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/metrics").status_code == 404
    # regimes returns [] for absent run, not 404.
    r = client.get("/api/runs/nope/regimes")
    assert r.status_code == 200 and r.json() == []


def test_post_run_and_full_rest_contract(tmp_path):
    client = _make_client(tmp_path)

    resp = client.post("/api/runs", json={"n_bars": 600, "seed": 3, "mode": "backtest"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert isinstance(run_id, str) and run_id

    run = _poll_until_persisted(client, run_id)
    assert run["run_id"] == run_id
    assert isinstance(run["steps"], list) and len(run["steps"]) > 0

    # /api/runs lists the new run.
    listed = client.get("/api/runs").json()["runs"]
    assert run_id in listed

    # regimes is a list of regime dicts.
    regimes = client.get(f"/api/runs/{run_id}/regimes").json()
    assert isinstance(regimes, list)
    assert all("regime_id" in r for r in regimes)

    # metrics has the documented keys / types.
    metrics = client.get(f"/api/runs/{run_id}/metrics").json()
    for key in (
        "n_steps",
        "n_trades",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "final_equity",
    ):
        assert key in metrics, key
    assert metrics["n_steps"] == len(run["steps"])
    assert isinstance(metrics["sharpe"], float)


def test_ws_stream_receives_events(tmp_path):
    """Drive the sync engine -> async WS bridge end to end."""
    client = _make_client(tmp_path)
    run_id = client.post("/api/runs", json={"n_bars": 500, "seed": 5}).json()["run_id"]

    received_types: list[str] = []
    with client.websocket_connect(f"/ws/stream?run_id={run_id}") as ws:
        while True:
            msg = ws.receive_json()
            received_types.append(msg["type"])
            if msg["type"] == "run_end":
                break

    # The stream must be well-formed: starts with run_start, ends with run_end,
    # and carries at least one step in between.
    assert received_types[0] == "run_start"
    assert received_types[-1] == "run_end"
    assert "step" in received_types


def test_ws_unknown_run_closes_gracefully(tmp_path):
    client = _make_client(tmp_path)
    with client.websocket_connect("/ws/stream?run_id=does-not-exist") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
