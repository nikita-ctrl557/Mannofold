"""Vectorized analytics over a run's StepResults.

This module COMPLEMENTS :func:`mannofold.engine.metrics.compute_metrics` (which
owns the headline scalars: sharpe, total_return, max_drawdown, win_rate, ...).
Nothing here re-implements those; instead it exposes time-series and grouped
views useful for diagnostics and the visualization layer.

Every function is pure and accepts either a sequence of
:class:`~mannofold.contracts.models.StepResult` or an already-flattened steps
``DataFrame`` (the layout produced by ``LocalStateStore._flatten`` /
``load_steps_df``). Outputs are plain ``dict`` / ``pandas`` objects.

Exposed analytics
-----------------
* ``rolling_sharpe(results, window)``  — rolling annualized Sharpe series.
* ``turnover(results)``                — gross/avg notional traded per step.
* ``per_regime_pnl(results)``          — PnL + hit-rate + count grouped by
  ``manifold.regime_id`` (counts reconcile to len(results)).
* ``exposure_stats(results)``          — avg/max gross & net exposure.
* ``drawdown_series(results)``         — equity, running peak, drawdown series.
* ``return_distribution(results)``     — mean/std/skew/kurtosis + VaR.
* ``analyze(results)``                 — bundles all of the above into one dict.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mannofold.contracts.models import StepResult

# Same convention as engine.metrics.compute_metrics: ~26 fifteen-minute bars/day.
PERIODS_PER_YEAR = 252 * 26

# Columns the flattened steps DataFrame is guaranteed to carry (see store._flatten).
_REQUIRED_COLS = (
    "seq",
    "regime_id",
    "target_weight",
    "equity",
    "drawdown",
    "net_exposure",
)


def to_frame(results: Sequence[StepResult] | pd.DataFrame) -> pd.DataFrame:
    """Normalize input to a flattened steps DataFrame.

    Accepts either StepResults or an already-flattened frame. The returned frame
    additionally carries ``returns`` and ``gross_exposure`` (not in the on-disk
    flatten layout) so the analytics below can run off either source.
    """
    if isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        df = pd.DataFrame(
            {
                "seq": [r.seq for r in results],
                "ts": [r.bar.ts for r in results],
                "regime_id": [r.manifold.regime_id for r in results],
                "target_weight": [r.target.target_weight for r in results],
                "equity": [r.portfolio.equity for r in results],
                "drawdown": [r.portfolio.drawdown for r in results],
                "net_exposure": [r.portfolio.net_exposure for r in results],
                "gross_exposure": [r.portfolio.gross_exposure for r in results],
                "returns": [r.portfolio.returns for r in results],
                "has_fill": [r.fill is not None for r in results],
            }
        )

    # Derive any missing complementary columns from what the flatten layout has.
    if "returns" not in df.columns:
        eq = df["equity"].to_numpy(dtype=float)
        ret = np.zeros_like(eq)
        if len(eq) > 1:
            ret[1:] = eq[1:] / np.where(eq[:-1] == 0.0, 1e-9, eq[:-1]) - 1.0
        df["returns"] = ret
    if "gross_exposure" not in df.columns:
        df["gross_exposure"] = df["net_exposure"].abs()
    if "has_fill" not in df.columns:
        # No fill information on disk; approximate via target_weight changes.
        df["has_fill"] = df["target_weight"].diff().fillna(0.0).abs() > 0
    return df


def rolling_sharpe(
    results: Sequence[StepResult] | pd.DataFrame,
    window: int = 64,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Rolling annualized Sharpe over per-step returns.

    Returns a DataFrame with ``seq`` and ``rolling_sharpe`` (NaN until the first
    full window). Annualization matches ``compute_metrics``.
    """
    df = to_frame(results)
    rets = df["returns"].astype(float)
    roll = rets.rolling(window, min_periods=window)
    mean = roll.mean()
    std = roll.std(ddof=0)
    sharpe = (mean / std.where(std > 0)) * np.sqrt(periods_per_year)
    out = pd.DataFrame({"seq": df["seq"].to_numpy(), "rolling_sharpe": sharpe.to_numpy()})
    return out


def turnover(results: Sequence[StepResult] | pd.DataFrame) -> dict:
    """Position turnover from |Δ target_weight| per step.

    ``total`` is summed absolute weight change; ``avg_per_step`` normalizes by the
    number of steps; ``n_trades`` counts steps that produced a fill.
    """
    df = to_frame(results)
    delta = df["target_weight"].astype(float).diff().abs().fillna(0.0)
    n = len(df)
    return {
        "total": float(delta.sum()),
        "avg_per_step": float(delta.sum() / n) if n else 0.0,
        "max_step": float(delta.max()) if n else 0.0,
        "n_trades": int(df["has_fill"].sum()),
    }


def per_regime_pnl(results: Sequence[StepResult] | pd.DataFrame) -> pd.DataFrame:
    """PnL, hit-rate and count grouped by ``manifold.regime_id``.

    Per step PnL is attributed as ``net_exposure * next-step return`` (the return
    realized while holding that exposure); the final step has no forward return
    and contributes 0. ``count`` sums to ``len(results)`` (reconciliation).
    Sorted by ``regime_id`` for stable output.
    """
    df = to_frame(results)
    rets = df["returns"].astype(float).to_numpy()
    expo = df["net_exposure"].astype(float).to_numpy()
    # PnL realized over the holding period = exposure_t * return_{t+1}.
    fwd_ret = np.zeros_like(rets)
    if len(rets) > 1:
        fwd_ret[:-1] = rets[1:]
    pnl = expo * fwd_ret

    g = pd.DataFrame({"regime_id": df["regime_id"].to_numpy(), "pnl": pnl, "ret": fwd_ret})
    rows = []
    for regime_id, sub in g.groupby("regime_id", sort=True):
        wins = int((sub["pnl"] > 0).sum())
        nonzero = int((sub["pnl"] != 0).sum())
        rows.append(
            {
                "regime_id": int(regime_id),
                "count": int(len(sub)),
                "pnl": float(sub["pnl"].sum()),
                "mean_pnl": float(sub["pnl"].mean()),
                "hit_rate": float(wins / nonzero) if nonzero else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["regime_id", "count", "pnl", "mean_pnl", "hit_rate"])


def exposure_stats(results: Sequence[StepResult] | pd.DataFrame) -> dict:
    """Average and maximum gross & net exposure across the run."""
    df = to_frame(results)
    gross = df["gross_exposure"].astype(float)
    net = df["net_exposure"].astype(float)
    n = len(df)
    return {
        "avg_gross": float(gross.mean()) if n else 0.0,
        "max_gross": float(gross.max()) if n else 0.0,
        "avg_net": float(net.mean()) if n else 0.0,
        "max_net": float(net.abs().max()) if n else 0.0,
        "avg_abs_net": float(net.abs().mean()) if n else 0.0,
    }


def drawdown_series(results: Sequence[StepResult] | pd.DataFrame) -> pd.DataFrame:
    """Equity, running peak and drawdown-from-peak as a time series.

    Recomputes the running peak from equity so the series is self-consistent even
    if reconstructed from disk.
    """
    df = to_frame(results)
    equity = df["equity"].astype(float)
    peak = equity.cummax()
    dd = equity / peak.where(peak != 0, 1e-9) - 1.0
    return pd.DataFrame(
        {
            "seq": df["seq"].to_numpy(),
            "equity": equity.to_numpy(),
            "peak": peak.to_numpy(),
            "drawdown": dd.to_numpy(),
        }
    )


def return_distribution(
    results: Sequence[StepResult] | pd.DataFrame, var_level: float = 0.05
) -> dict:
    """Moments of the per-step return distribution + historical VaR/CVaR.

    ``skew``/``kurtosis`` are sample estimators (excess kurtosis). ``var`` is the
    ``var_level`` empirical quantile (a negative number for a loss); ``cvar`` is
    the mean of returns at or below that quantile.
    """
    df = to_frame(results)
    r = df["returns"].astype(float).to_numpy()
    n = r.size
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "var": 0.0,
            "cvar": 0.0,
            "var_level": var_level,
            "n": 0,
        }
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    if std > 0 and n > 0:
        z = (r - mean) / std
        skew = float((z**3).mean())
        kurtosis = float((z**4).mean() - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0
    var = float(np.quantile(r, var_level))
    tail = r[r <= var]
    cvar = float(tail.mean()) if tail.size else var
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "var": var,
        "cvar": cvar,
        "var_level": var_level,
        "n": int(n),
    }


def analyze(
    results: Sequence[StepResult] | pd.DataFrame,
    rolling_window: int = 64,
    var_level: float = 0.05,
) -> dict:
    """Bundle every analytic into one dict.

    DataFrame-valued analytics are returned as records (``list[dict]``) so the
    result is JSON-serializable for the API/frontend. ``n_steps`` is included for
    quick reconciliation against ``per_regime_pnl`` counts.
    """
    df = to_frame(results)
    regime = per_regime_pnl(df)
    return {
        "n_steps": int(len(df)),
        "rolling_sharpe": rolling_sharpe(df, window=rolling_window).to_dict(orient="records"),
        "turnover": turnover(df),
        "per_regime_pnl": regime.to_dict(orient="records"),
        "exposure_stats": exposure_stats(df),
        "drawdown_series": drawdown_series(df).to_dict(orient="records"),
        "return_distribution": return_distribution(df, var_level=var_level),
    }
