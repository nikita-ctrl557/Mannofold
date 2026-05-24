"""Rolling feature pipeline with an embedded scaler.

The scaler is fit ONLY inside :meth:`fit` (train window) and merely applied in
:meth:`transform`. Keeping it inside the pipeline means no caller can accidentally
fit it on the full series — the single most common source of lookahead bias.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from mannofold.contracts.models import Bar, FeatureVector
from mannofold.features import indicators as ind

_FEATURE_NAMES = [
    "ret_1",
    "mom_5",
    "mom_10",
    "mom_20",
    "accel_10",
    "vol_10",
    "vol_20",
    "rsi_14",
    "sma_ratio_20",
    "range_pct",
    "volume_z_20",
]

WARMUP = 41  # need 40 trailing bars + current for the longest lookback


class RollingFeaturePipeline:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    @property
    def warmup(self) -> int:
        return WARMUP

    @property
    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)

    def _raw(self, window: Sequence[Bar]) -> np.ndarray:
        close = np.array([b.close for b in window], dtype=float)
        high = window[-1].high
        low = window[-1].low
        volume = np.array([b.volume for b in window], dtype=float)
        r = ind.log_returns(close)
        return np.array(
            [
                float(r[-1]) if len(r) else 0.0,
                ind.momentum(close, 5),
                ind.momentum(close, 10),
                ind.momentum(close, 20),
                ind.acceleration(close, 10),
                ind.realized_vol(close, 10),
                ind.realized_vol(close, 20),
                ind.rsi(close, 14) / 100.0,
                ind.sma_ratio(close, 20),
                ind.range_pct(high, low, window[-1].close),
                ind.volume_z(volume, 20),
            ],
            dtype=float,
        )

    def _raw_matrix(self, bars: Sequence[Bar]) -> np.ndarray:
        rows = [
            self._raw(bars[i - self.warmup + 1 : i + 1])
            for i in range(self.warmup - 1, len(bars))
        ]
        return np.asarray(rows, dtype=float)

    def fit(self, bars: Sequence[Bar]) -> None:
        if len(bars) < self.warmup:
            raise ValueError(f"need >= {self.warmup} bars to fit, got {len(bars)}")
        raw = self._raw_matrix(bars)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self._scaler.fit(raw)
        self._fitted = True

    def transform(self, window: Sequence[Bar]) -> FeatureVector:
        if not self._fitted:
            raise RuntimeError("RollingFeaturePipeline.transform called before fit")
        if len(window) < self.warmup:
            raise ValueError(f"window too short: {len(window)} < {self.warmup}")
        raw = self._raw(window[-self.warmup :]).reshape(1, -1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self._scaler.transform(raw)[0]
        return FeatureVector(
            ts=window[-1].ts,
            symbol=window[-1].symbol,
            values=scaled.tolist(),
            names=self.feature_names,
        )
