# Mannofold developer tasks. All Python targets use the project's .venv directly
# (never uv/pip at runtime), matching the CI install.

PY      := .venv/bin/python
PYTEST  := .venv/bin/pytest
RUFF    := .venv/bin/ruff
UVICORN := .venv/bin/uvicorn

.DEFAULT_GOAL := help

.PHONY: help install synth backtest test lint api web demo dev

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the project with dev + manifold extras into .venv.
	uv pip install -e ".[dev,manifold]"

synth:  ## Generate a synthetic regime-switching dataset (Parquet).
	$(PY) scripts/gen_synthetic.py

backtest:  ## Run a backtest on synthetic data and persist run artifacts.
	$(PY) scripts/run_backtest.py

test:  ## Run the test suite.
	$(PYTEST) -q

lint:  ## Lint with ruff.
	$(RUFF) check .

api:  ## Serve the FastAPI app on :8000.
	$(UVICORN) mannofold.api.app:app --port 8000

web:  ## Run the Vite dev server in web/.
	npm --prefix web run dev

demo: synth backtest api  ## Generate data, run a backtest, then serve the API.

dev:  ## One command: set up, seed data, run API + dashboard at localhost:5173.
	./scripts/run_local.sh
