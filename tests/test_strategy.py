"""Unit tests for the churn-curbing strategy + risk sizer.

These pin the contract behaviours the engine relies on:

* a positive expected return + high confidence + low anomaly produces a positive
  target weight (once it clears the entry threshold);
* a highly anomalous state is de-grossed to a small magnitude;
* the target weight always stays bounded in ``[-1, 1]``;
* :class:`VolTargetRiskSizer` returns ``None`` inside its rebalance band and an
  :class:`Order` once the gap clears it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mannofold.contracts.models import (
    ANOMALY_REGIME,
    PortfolioState,
    Side,
    SignalSet,
    TargetPosition,
)
from mannofold.signals.risk import VolTargetRiskSizer
from mannofold.signals.strategy import ManifoldStrategy

TS = datetime(2024, 1, 1, tzinfo=UTC)


def _signals(**kw) -> SignalSet:
    base = dict(
        ts=TS,
        symbol="SYNTH",
        momentum=0.0,
        expected_return=0.0,
        anomaly=0.0,
        regime_id=0,
        confidence=0.0,
    )
    base.update(kw)
    return SignalSet(**base)


def test_positive_setup_yields_positive_weight():
    strat = ManifoldStrategy()
    # Strong positive edge, fully confident, on-manifold, identified regime.
    sig = _signals(expected_return=0.02, confidence=0.9, anomaly=0.0, regime_id=1)
    tgt = strat.target(sig)
    assert tgt.target_weight > 0.0


def test_high_anomaly_is_degrossed():
    strat = ManifoldStrategy()
    sig = _signals(expected_return=0.05, confidence=0.9, anomaly=0.95, regime_id=1)
    tgt = strat.target(sig)
    assert abs(tgt.target_weight) < 0.1


def test_off_manifold_regime_blocks_entry():
    strat = ManifoldStrategy()
    sig = _signals(
        expected_return=0.05, confidence=0.9, anomaly=0.1, regime_id=ANOMALY_REGIME
    )
    assert strat.target(sig).target_weight == 0.0


def test_weight_bounded():
    strat = ManifoldStrategy(gain=1e6)
    for er in (-10.0, -0.1, 0.1, 10.0):
        sig = _signals(expected_return=er, confidence=1.0, anomaly=0.0, regime_id=1)
        w = strat.target(sig).target_weight
        assert -1.0 <= w <= 1.0


def test_hysteresis_holds_then_exits():
    strat = ManifoldStrategy()
    # Build up a position with a strong, repeated signal.
    strong = _signals(expected_return=0.03, confidence=0.9, anomaly=0.0, regime_id=1)
    for _ in range(10):
        w = strat.target(strong).target_weight
    assert w > 0.0
    # Decay conviction below the exit threshold -> position is released to flat.
    weak = _signals(expected_return=0.0, confidence=0.9, anomaly=0.0, regime_id=1)
    for _ in range(10):
        w = strat.target(weak).target_weight
    assert w == 0.0


def test_low_confidence_blocks_entry():
    strat = ManifoldStrategy()
    sig = _signals(expected_return=0.05, confidence=0.05, anomaly=0.0, regime_id=1)
    assert strat.target(sig).target_weight == 0.0


def _portfolio(qty: float, price: float, equity: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        ts=TS,
        cash=equity - qty * price,
        equity=equity,
        positions={"SYNTH": qty},
    )


def test_risk_returns_none_inside_band():
    risk = VolTargetRiskSizer(target_vol=0.01, rebalance_band=0.05)
    price = 100.0
    # Target maps (after vol scaling 0.01/0.01 = 1.0) to weight ~0.30 -> $30k.
    target = TargetPosition(ts=TS, symbol="SYNTH", target_weight=0.30)
    # Current value already ~$30k (300 * 100): delta tiny -> inside band -> None.
    order = risk.size(target, _portfolio(300.0, price), price, anomaly=0.0, volatility=0.01)
    assert order is None


def test_risk_returns_order_outside_band():
    risk = VolTargetRiskSizer(target_vol=0.01, rebalance_band=0.05)
    price = 100.0
    target = TargetPosition(ts=TS, symbol="SYNTH", target_weight=0.50)
    # Flat book, desired ~$50k of exposure -> well outside the band -> a BUY.
    order = risk.size(target, _portfolio(0.0, price), price, anomaly=0.0, volatility=0.01)
    assert order is not None
    assert order.side == Side.BUY
    assert order.qty > 0.0


def test_risk_degrosses_on_anomaly():
    risk = VolTargetRiskSizer(target_vol=0.01, rebalance_band=0.05)
    price = 100.0
    target = TargetPosition(ts=TS, symbol="SYNTH", target_weight=0.50)
    # High anomaly collapses desired exposure; from a flat book that is inside
    # the band, so no order is produced.
    order = risk.size(target, _portfolio(0.0, price), price, anomaly=0.99, volatility=0.01)
    assert order is None
