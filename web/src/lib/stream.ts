import type { StreamEvent, StepResult, Regime } from "../types/contracts";

export interface StreamHandlers {
  onStep?: (step: StepResult, seq: number) => void;
  onRegimes?: (regimes: Regime[]) => void;
  onStart?: (runId: string) => void;
  onEnd?: () => void;
  onStatus?: (s: string) => void;
}

// POST a new run then connect the WS and dispatch decoded events.
// Tolerates seq gaps (server uses drop-oldest backpressure).
export async function startLiveRun(
  handlers: StreamHandlers
): Promise<{ close: () => void }> {
  handlers.onStatus?.("starting run...");
  let runId = "";
  try {
    const r = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    if (r.ok) {
      const j = (await r.json()) as { run_id: string };
      runId = j.run_id;
    }
  } catch {
    // ignore; runId may be empty and the WS will use a default
  }

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  const ws = new WebSocket(`${proto}//${location.host}/ws/stream${qs}`);

  let lastSeq = -1;

  ws.onopen = () => handlers.onStatus?.("connected");
  ws.onerror = () => handlers.onStatus?.("ws error");
  ws.onclose = () => {
    handlers.onStatus?.("disconnected");
    handlers.onEnd?.();
  };
  ws.onmessage = (ev) => {
    let msg: StreamEvent;
    try {
      msg = JSON.parse(ev.data as string) as StreamEvent;
    } catch {
      return;
    }
    if (typeof msg.seq === "number") {
      if (msg.seq < lastSeq) return; // out of order; ignore
      lastSeq = msg.seq;
    }
    switch (msg.type) {
      case "run_start":
        handlers.onStart?.(msg.run_id);
        break;
      case "step":
        handlers.onStep?.(msg.payload as unknown as StepResult, msg.seq);
        break;
      case "regime_fit": {
        const p = msg.payload as { regimes?: Regime[] };
        if (p.regimes) handlers.onRegimes?.(p.regimes);
        break;
      }
      case "run_end":
        handlers.onEnd?.();
        break;
      default:
        break;
    }
  };

  return {
    close: () => {
      try {
        ws.close();
      } catch {
        /* noop */
      }
    },
  };
}
