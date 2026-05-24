"""Broker adapters — execute orders against an external venue.

``AlpacaPaperBroker`` submits orders to the Alpaca paper-trading REST API behind
the same domain types (:class:`Order` in, :class:`Fill` out) the engine uses.
"""

from mannofold.broker.alpaca_broker import AlpacaPaperBroker

__all__ = ["AlpacaPaperBroker"]
