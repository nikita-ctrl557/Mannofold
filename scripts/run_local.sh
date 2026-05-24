#!/usr/bin/env bash
# One-command local boot for the Mannofold dashboard:
#   - creates the Python venv + installs deps (first run only)
#   - seeds real VIX + a synthetic run if no data exists yet
#   - starts the API on :8000 and the dashboard on http://localhost:5173
#
# Usage (from the repo root):  ./scripts/run_local.sh   (or: make dev)
set -euo pipefail
cd "$(dirname "$0")/.."

command -v python3 >/dev/null || { echo "ERROR: python3 not found (brew install python@3.11)"; exit 1; }
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' \
  || { echo "ERROR: Mannofold needs Python 3.11+ (brew install python@3.11)"; exit 1; }
command -v node >/dev/null || { echo "ERROR: Node 18+ not found (brew install node)"; exit 1; }

if [ ! -d .venv ]; then
  echo "[mannofold] creating .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[mannofold] installing python deps (first run is slowest) ..."
pip install -q --upgrade pip
pip install -q -e ".[dev]"

if [ ! -d data/runs/vix ]; then
  echo "[mannofold] seeding data: real VIX history + a synthetic run ..."
  python scripts/fetch_historical.py vix vix
  python scripts/run_backtest.py synthetic
fi

echo "[mannofold] starting API on http://localhost:8000 ..."
python -m uvicorn mannofold.api.app:app --port 8000 &
API_PID=$!
trap 'echo; echo "[mannofold] stopping API ($API_PID) ..."; kill "$API_PID" 2>/dev/null || true' EXIT INT TERM

cd web
[ -d node_modules ] || { echo "[mannofold] installing web deps ..."; npm install; }
echo "[mannofold] dashboard → http://localhost:5173   (Ctrl+C to stop both)"
npm run dev
