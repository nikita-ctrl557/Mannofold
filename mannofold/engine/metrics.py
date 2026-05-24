"""Performance metrics computed from a run's StepResults."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mannofold.contracts.models import StepResult


def compute_metrics(results: Sequence[StepResult], periods_per_year: int = 252 * 26) -> dict:
    """``periods_per_year`` default assumes ~26 fifteen-minute bars per trading day."""
    if not results:
        return {}
    rets = np.array([r.portfolio.returns for r in results], dtype=float)
    equity = np.array([r.portfolio.equity for r in results], dtype=float)
    drawdowns = np.array([r.portfolio.drawdown for r in results], dtype=float)
    fills = [r for r in results if r.fill is not None]

    mean, std = float(rets.mean()), float(rets.std())
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std > 0 else 0.0
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] else 0.0
    wins = [r for r in results if r.portfolio.returns > 0]

    return {
        "n_steps": len(results),
        "n_trades": len(fills),
        "total_return": total_return,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdowns.min()) if len(drawdowns) else 0.0,
        "win_rate": len(wins) / len(results) if results else 0.0,
        "final_equity": float(equity[-1]),
    }
