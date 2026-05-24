"""Volatility-targeting risk sizer with anomaly de-grossing and a rebalance band.

Churn is curbed with a two-level rebalance band (hysteresis): once a position is
open, a *wider* band keeps it from being nudged by small target drifts; only a
move that clears the (wider) band, or a move toward flat, triggers an order. A
minimum-trade floor suppresses dust orders. Anomaly de-grossing is preserved.

All knobs are constructor args with defaults, so ``VolTargetRiskSizer()`` still
constructs with no arguments. The sizer is a pure function of its inputs.
"""

from __future__ import annotations

from mannofold.contracts.models import Order, PortfolioState, Side, TargetPosition

_EPS = 1e-9


class VolTargetRiskSizer:
    def __init__(
        self,
        target_vol: float = 0.01,
        max_leverage: float = 1.0,
        rebalance_band: float = 0.05,
        commission_bps: float = 1.0,
        hold_band_mult: float = 2.0,
        min_trade_frac: float = 0.01,
    ):
        self._target_vol = target_vol
        self._max_leverage = max_leverage
        self._band = rebalance_band
        self._commission_bps = commission_bps
        # While holding a same-side position, require a larger gap before
        # rebalancing (hysteresis) to avoid trading on small target wobble.
        self._hold_band_mult = max(hold_band_mult, 1.0)
        # Suppress dust trades below this fraction of equity.
        self._min_trade_frac = max(min_trade_frac, 0.0)

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

        equity = max(portfolio.equity, _EPS)
        desired_value = weight * portfolio.equity
        current_qty = portfolio.positions.get(target.symbol, 0.0)
        current_value = current_qty * price
        delta_value = desired_value - current_value

        # Hysteresis: only widen the band when we'd be *adding* same-side risk.
        # Trimming toward flat (or flipping) uses the tighter base band so risk
        # can always be reduced promptly.
        holding = abs(current_value) > _EPS
        adding = holding and (desired_value >= 0.0) == (current_value >= 0.0) and abs(
            desired_value
        ) > abs(current_value)
        band = self._band * (self._hold_band_mult if adding else 1.0)

        if abs(delta_value) < band * equity:
            return None
        if abs(delta_value) < self._min_trade_frac * equity:
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
