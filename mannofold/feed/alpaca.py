"""Alpaca paper-trading market-data feed.

``AlpacaFeed`` implements the :class:`~mannofold.contracts.interfaces.DataFeed`
Protocol (``mode == Mode.PAPER``) by fetching historical OHLCV bars from the
Alpaca Market Data v2 REST API and yielding them as domain :class:`Bar` objects
in chronological order.

Live use requires:

* Network egress to ``*.alpaca.markets`` (this sandbox blocks it — see error
  text below; allowlist the host or run locally).
* Credentials in the env vars ``MANNOFOLD_ALPACA_KEY`` / ``MANNOFOLD_ALPACA_SECRET``
  (paper keys start with ``PK``). These are sent as request headers and are
  NEVER logged.

The constructor accepts an injectable ``httpx.Client`` so tests can drive the
adapter with an :class:`httpx.MockTransport` and never touch the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime

import httpx

from mannofold.contracts.models import Bar, Mode

KEY_ENV = "MANNOFOLD_ALPACA_KEY"
SECRET_ENV = "MANNOFOLD_ALPACA_SECRET"
DEFAULT_DATA_URL = "https://data.alpaca.markets"


def _iso(value: datetime | str) -> str:
    """Normalize a start/end bound to an ISO-8601 string."""
    return value.isoformat() if isinstance(value, datetime) else value


class AlpacaFeed:
    """Historical bar feed backed by Alpaca Market Data v2.

    Parameters
    ----------
    symbol:
        Ticker, e.g. ``"AAPL"``.
    start, end:
        Inclusive time bounds (``datetime`` or ISO-8601 string).
    timeframe:
        Alpaca bar timeframe, e.g. ``"15Min"``, ``"1Day"``.
    limit:
        Max bars per page (Alpaca caps at 10000); pagination is automatic.
    base_url:
        Override for the data API host (tests point this at a mock).
    client:
        Optional injected :class:`httpx.Client` (e.g. wrapping a
        :class:`httpx.MockTransport`). When omitted a real client is built
        lazily at stream time.
    """

    mode: Mode = Mode.PAPER

    def __init__(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        timeframe: str = "15Min",
        limit: int = 1000,
        base_url: str = DEFAULT_DATA_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.symbol = symbol
        self.start = _iso(start)
        self.end = _iso(end)
        self.timeframe = timeframe
        self.limit = limit
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(KEY_ENV)
        secret = os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise ValueError(
                f"Alpaca credentials missing: set {KEY_ENV} and {SECRET_ENV} "
                "in the environment (paper keys start with 'PK'). "
                "Credentials are sent as headers and never logged."
            )
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    @staticmethod
    def _to_bar(raw: dict, symbol: str) -> Bar:
        """Map an Alpaca bar ``{t,o,h,l,c,v}`` to a domain :class:`Bar`."""
        return Bar(
            ts=raw["t"],
            symbol=symbol,
            open=raw["o"],
            high=raw["h"],
            low=raw["l"],
            close=raw["c"],
            volume=raw["v"],
        )

    def stream(self) -> Iterator[Bar]:
        """Yield every bar in ``[start, end]`` in chronological order.

        Pages through the API using ``next_page_token`` until exhausted.
        """
        headers = self._headers()
        client = self._client or httpx.Client(base_url=self.base_url, timeout=30.0)
        owns_client = self._client is None
        path = f"/v2/stocks/{self.symbol}/bars"
        params: dict[str, str | int] = {
            "timeframe": self.timeframe,
            "start": self.start,
            "end": self.end,
            "limit": self.limit,
        }
        try:
            page_token: str | None = None
            while True:
                page_params = dict(params)
                if page_token:
                    page_params["page_token"] = page_token
                try:
                    resp = client.get(path, params=page_params, headers=headers)
                except httpx.ConnectError as exc:  # blocked / unreachable host
                    raise ConnectionError(
                        f"Cannot reach Alpaca at {self.base_url}: {exc}. "
                        "Allowlist *.alpaca.markets or run locally."
                    ) from exc
                resp.raise_for_status()
                payload = resp.json()
                for raw in payload.get("bars") or []:
                    yield self._to_bar(raw, self.symbol)
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        finally:
            if owns_client:
                client.close()

    def live_stream(self) -> Iterator[Bar]:  # pragma: no cover - documented stub
        """Real-time bar feed via the Alpaca WebSocket stream (not implemented).

        The production implementation would connect to
        ``wss://stream.data.alpaca.markets/v2/iex``, authenticate with the same
        env credentials, subscribe to ``bars`` for ``self.symbol``, and yield a
        :class:`Bar` per ``{"T": "b", ...}` message. Requires egress to
        ``*.alpaca.markets`` (allowlist or run locally).
        """
        raise NotImplementedError(
            "AlpacaFeed.live_stream (WebSocket) is not implemented in this build; "
            "use stream() for historical bars. Live use needs egress to "
            "*.alpaca.markets (allowlist or run locally)."
        )
