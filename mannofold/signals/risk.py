"""Volatility-targeting risk sizer with anomaly de-grossing and a rebalance band."""

from __future__ import annotations

from mannofold.contracts.models import Order, PortfolioState, Side, TargetPosition

_EPS = 1e-9


class VolTargetRiskSizer:
    def __init__(
        self,
        target_vol: float = 0.01,
        max_leverage: float = 1.0,
        rebalance_band: float = 0.02,
        commission_bps: float = 1.0,
    ):
        self._target_vol = target_vol
        self._max_leverage = max_leverage
        self._band = rebalance_band
        self._commission_bps = commission_bps

    def size(
        self,
        target: TargetPosition,
        portfolio: PortfolioState,
        price: float,
        anomaly: float,
        volatility: float,
    ) -> Order | None:
        # Scale exposure so realized vol ≈ target vol, then cap leverage.
        scale = self._target_vol / (volatility + _EPS)
        weight = target.target_weight * scale
        weight = max(-self._max_leverage, min(self._max_leverage, weight))
        weight *= 1.0 - anomaly  # extra de-gross off-manifold

        desired_value = weight * portfolio.equity
        current_qty = portfolio.positions.get(target.symbol, 0.0)
        current_value = current_qty * price
        delta_value = desired_value - current_value

        if abs(delta_value) < self._band * max(portfolio.equity, _EPS):
            return None

        qty = delta_value / (price + _EPS)
        side = Side.BUY if qty > 0 else Side.SELL
        return Order(
            ts=target.ts,
            symbol=target.symbol,
            side=side,
            qty=abs(qty),
            target_weight=weight,
            reason=f"rebalance to w={weight:.3f}",
        )
