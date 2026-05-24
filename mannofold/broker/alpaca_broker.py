"""Alpaca paper-trading broker adapter.

``AlpacaPaperBroker`` turns a domain :class:`Order` into an Alpaca market order
(``POST /v2/orders``) and maps the response back to a :class:`Fill`. It also
exposes account + position reads.

Live use requires:

* Network egress to ``paper-api.alpaca.markets`` (blocked in this sandbox —
  allowlist ``*.alpaca.markets`` or run locally).
* Credentials in ``MANNOFOLD_ALPACA_KEY`` / ``MANNOFOLD_ALPACA_SECRET`` (paper
  keys start with ``PK``). Sent as headers, NEVER logged.

Tests inject an :class:`httpx.Client` (typically wrapping an
:class:`httpx.MockTransport`) so no real network call is made.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from mannofold.contracts.models import Fill, Order, Side

KEY_ENV = "MANNOFOLD_ALPACA_KEY"
SECRET_ENV = "MANNOFOLD_ALPACA_SECRET"
DEFAULT_PAPER_URL = "https://paper-api.alpaca.markets"


class AlpacaPaperBroker:
    """Submit/inspect orders against the Alpaca paper-trading API."""

    def __init__(
        self,
        base_url: str = DEFAULT_PAPER_URL,
        client: httpx.Client | None = None,
    ) -> None:
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

    def _client_or_default(self) -> tuple[httpx.Client, bool]:
        if self._client is not None:
            return self._client, False
        return httpx.Client(base_url=self.base_url, timeout=30.0), True

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        client, owns = self._client_or_default()
        try:
            try:
                resp = client.request(method, path, headers=self._headers(), **kwargs)
            except httpx.ConnectError as exc:
                raise ConnectionError(
                    f"Cannot reach Alpaca at {self.base_url}: {exc}. "
                    "Allowlist *.alpaca.markets or run locally."
                ) from exc
            resp.raise_for_status()
            return resp.json()
        finally:
            if owns:
                client.close()

    def submit(self, order: Order) -> Fill:
        """Place a market order and map Alpaca's response to a :class:`Fill`."""
        body = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.value,
            "type": "market",
            "time_in_force": "day",
        }
        data = self._request("POST", "/v2/orders", json=body)
        assert isinstance(data, dict)
        return self._to_fill(data, order)

    @staticmethod
    def _to_fill(data: dict, order: Order) -> Fill:
        """Map an Alpaca order/fill response onto a domain :class:`Fill`.

        Falls back to the submitted order's fields when Alpaca returns an
        accepted-but-not-yet-filled order (price/qty may be null).
        """
        filled_qty = data.get("filled_qty")
        qty = float(filled_qty) if filled_qty not in (None, "") else order.qty

        avg_price = data.get("filled_avg_price")
        price = float(avg_price) if avg_price not in (None, "") else 0.0

        side_raw = data.get("side") or order.side.value
        try:
            side = Side(side_raw)
        except ValueError:
            side = order.side

        ts_raw = data.get("filled_at") or data.get("submitted_at")
        ts = ts_raw if ts_raw else datetime.now(UTC)

        return Fill(
            ts=ts,
            symbol=data.get("symbol", order.symbol),
            side=side,
            qty=qty,
            price=price,
        )

    def account(self) -> dict:
        """Return the raw Alpaca account object."""
        data = self._request("GET", "/v2/account")
        assert isinstance(data, dict)
        return data

    def positions(self) -> list:
        """Return the list of open positions."""
        data = self._request("GET", "/v2/positions")
        assert isinstance(data, list)
        return data
