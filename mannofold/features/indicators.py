"""Pure, causal technical indicators operating on trailing arrays.

Every function takes arrays whose LAST element is the current bar and returns a
scalar computed from that point looking only backwards — no future leakage.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.clip(close, EPS, None)))


def momentum(close: np.ndarray, horizon: int) -> float:
    if len(close) <= horizon:
        return 0.0
    return float(np.log(close[-1] + EPS) - np.log(close[-1 - horizon] + EPS))


def realized_vol(close: np.ndarray, window: int) -> float:
    r = log_returns(close[-(window + 1) :])
    return float(np.std(r)) if len(r) else 0.0


def rsi(close: np.ndarray, window: int = 14) -> float:
    r = np.diff(close[-(window + 1) :])
    if len(r) == 0:
        return 50.0
    gains = np.clip(r, 0, None).mean()
    losses = -np.clip(r, None, 0).mean()
    rs = gains / (losses + EPS)
    return float(100.0 - 100.0 / (1.0 + rs))


def sma_ratio(close: np.ndarray, window: int) -> float:
    sma = float(np.mean(close[-window:]))
    return float(close[-1] / (sma + EPS) - 1.0)


def range_pct(high: float, low: float, close: float) -> float:
    return float((high - low) / (close + EPS))


def volume_z(volume: np.ndarray, window: int) -> float:
    v = volume[-window:]
    mu = float(np.mean(v))
    sd = float(np.std(v))
    return float((volume[-1] - mu) / (sd + EPS))


def acceleration(close: np.ndarray, horizon: int) -> float:
    """Momentum-of-momentum: change in short-horizon momentum."""
    if len(close) <= 2 * horizon:
        return 0.0
    m_now = momentum(close, horizon)
    m_prev = momentum(close[: -horizon], horizon)
    return float(m_now - m_prev)
