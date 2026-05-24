"""FastAPI app exposing the Mannofold engine over REST + WebSocket.

Run with::

    .venv/bin/python -m uvicorn mannofold.api.app:app --port 8000
"""

from mannofold.api.app import app, create_app

__all__ = ["app", "create_app"]
