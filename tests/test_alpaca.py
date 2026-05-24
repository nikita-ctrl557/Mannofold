"""Unit tests for the Alpaca paper-trading adapters.

All network I/O is mocked with :class:`httpx.MockTransport` — no real calls are
made to ``*.alpaca.markets`` (which is also firewalled in CI). Credentials are
injected via monkeypatched env vars and asserted to flow through as headers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from mannofold.broker import AlpacaPaperBroker
from mannofold.contracts.interfaces import DataFeed
from mannofold.contracts.models import Bar, Fill, Mode, Order, Side
from mannofold.feed import AlpacaFeed

KEY = "PKTEST1234567890"
SECRET = "secret-shhh"

# Two pages of bars; page 1 carries a next_page_token, page 2 ends it.
PAGE_1 = {
    "bars": [
        {"t": "2023-01-02T14:30:00Z", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1200},
        {"t": "2023-01-02T14:45:00Z", "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.8, "v": 1500},
    ],
    "symbol": "AAPL",
    "next_page_token": "tok-2",
}
PAGE_2 = {
    "bars": [
        {"t": "2023-01-02T15:00:00Z", "o": 101.8, "h": 103.0, "l": 101.0, "c": 102.4, "v": 1700},
    ],
    "symbol": "AAPL",
    "next_page_token": None,
}


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("MANNOFOLD_ALPACA_KEY", KEY)
    monkeypatch.setenv("MANNOFOLD_ALPACA_SECRET", SECRET)


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://mock")


def test_feed_is_datafeed_protocol_and_paper_mode():
    feed = AlpacaFeed("AAPL", "2023-01-02", "2023-01-03", client=_mock_client(lambda r: httpx.Response(200, json=PAGE_2)))
    assert isinstance(feed, DataFeed)
    assert feed.mode is Mode.PAPER
    assert AlpacaFeed.mode is Mode.PAPER


def test_feed_paginates_and_yields_typed_bars_in_order():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        # Header credentials must be forwarded.
        assert request.headers["APCA-API-KEY-ID"] == KEY
        assert request.headers["APCA-API-SECRET-KEY"] == SECRET
        assert request.url.path == "/v2/stocks/AAPL/bars"
        token = request.url.params.get("page_token")
        return httpx.Response(200, json=PAGE_2 if token == "tok-2" else PAGE_1)

    feed = AlpacaFeed(
        "AAPL",
        start=datetime(2023, 1, 2, tzinfo=UTC),
        end=datetime(2023, 1, 3, tzinfo=UTC),
        timeframe="15Min",
        client=_mock_client(handler),
    )
    bars = list(feed.stream())

    assert len(bars) == 3  # 2 from page 1 + 1 from page 2
    assert all(isinstance(b, Bar) for b in bars)
    assert [b.ts for b in bars] == sorted(b.ts for b in bars)
    first = bars[0]
    assert first.symbol == "AAPL"
    assert (first.open, first.high, first.low, first.close, first.volume) == (
        100.0,
        101.0,
        99.5,
        100.5,
        1200.0,
    )
    # First page request carried no page_token; second one did.
    assert "page_token" not in calls[0]
    assert calls[1]["page_token"] == "tok-2"
    assert calls[0]["timeframe"] == "15Min"


def test_feed_empty_payload_yields_nothing():
    feed = AlpacaFeed(
        "AAPL", "2023-01-02", "2023-01-03",
        client=_mock_client(lambda r: httpx.Response(200, json={"bars": None, "next_page_token": None})),
    )
    assert list(feed.stream()) == []


def test_feed_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("MANNOFOLD_ALPACA_KEY", raising=False)
    monkeypatch.delenv("MANNOFOLD_ALPACA_SECRET", raising=False)
    feed = AlpacaFeed("AAPL", "2023-01-02", "2023-01-03", client=_mock_client(lambda r: httpx.Response(200, json=PAGE_2)))
    with pytest.raises(ValueError, match="credentials missing"):
        list(feed.stream())


def test_live_stream_documented_stub_raises():
    feed = AlpacaFeed("AAPL", "2023-01-02", "2023-01-03", client=_mock_client(lambda r: httpx.Response(200, json=PAGE_2)))
    with pytest.raises(NotImplementedError, match="alpaca.markets"):
        next(feed.live_stream())


def test_broker_submit_returns_fill():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/orders"
        assert request.headers["APCA-API-KEY-ID"] == KEY
        import json

        body = json.loads(request.content)
        assert body == {
            "symbol": "AAPL",
            "qty": "10.0",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        return httpx.Response(
            200,
            json={
                "id": "order-1",
                "symbol": "AAPL",
                "side": "buy",
                "filled_qty": "10",
                "filled_avg_price": "101.25",
                "filled_at": "2023-01-02T14:31:00Z",
            },
        )

    broker = AlpacaPaperBroker(client=_mock_client(handler))
    order = Order(ts=datetime(2023, 1, 2, tzinfo=UTC), symbol="AAPL", side=Side.BUY, qty=10.0)
    fill = broker.submit(order)

    assert isinstance(fill, Fill)
    assert fill.symbol == "AAPL"
    assert fill.side is Side.BUY
    assert fill.qty == 10.0
    assert fill.price == 101.25


def test_broker_submit_unfilled_falls_back_to_order():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "order-2",
                "symbol": "AAPL",
                "side": "sell",
                "filled_qty": None,
                "filled_avg_price": None,
                "submitted_at": "2023-01-02T14:31:00Z",
            },
        )

    broker = AlpacaPaperBroker(client=_mock_client(handler))
    order = Order(ts=datetime(2023, 1, 2, tzinfo=UTC), symbol="AAPL", side=Side.SELL, qty=5.0)
    fill = broker.submit(order)
    assert fill.qty == 5.0  # fell back to submitted qty
    assert fill.price == 0.0
    assert fill.side is Side.SELL


def test_broker_account_and_positions():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            return httpx.Response(200, json={"cash": "100000", "status": "ACTIVE"})
        return httpx.Response(200, json=[{"symbol": "AAPL", "qty": "10"}])

    broker = AlpacaPaperBroker(client=_mock_client(handler))
    assert broker.account()["status"] == "ACTIVE"
    positions = broker.positions()
    assert isinstance(positions, list) and positions[0]["symbol"] == "AAPL"


def test_broker_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("MANNOFOLD_ALPACA_KEY", raising=False)
    monkeypatch.delenv("MANNOFOLD_ALPACA_SECRET", raising=False)
    broker = AlpacaPaperBroker(client=_mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ValueError, match="credentials missing"):
        broker.account()
