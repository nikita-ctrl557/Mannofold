"""RollingFeaturePipeline behaviour: shape, guards, finiteness, determinism.

The scaler is fit-only inside :meth:`fit`; these tests pin that invariant plus
the warmup/ordering contracts the engine relies on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mannofold.features.pipeline import RollingFeaturePipeline
from mannofold.feed.synthetic import SyntheticConfig, generate_bars


def _bars(n: int = 600, seed: int = 3):
    bars, _ = generate_bars(SyntheticConfig(n_bars=n, seed=seed))
    return bars


def test_output_length_matches_feature_names():
    bars = _bars()
    pipe = RollingFeaturePipeline()
    pipe.fit(bars)
    fv = pipe.transform(bars[-pipe.warmup :])
    assert len(fv.values) == len(pipe.feature_names)
    assert fv.names == pipe.feature_names
    assert len(pipe.feature_names) == len(set(pipe.feature_names))


def test_transform_before_fit_raises():
    pipe = RollingFeaturePipeline()
    bars = _bars()
    with pytest.raises(RuntimeError):
        pipe.transform(bars[-pipe.warmup :])


def test_fit_too_few_bars_raises():
    pipe = RollingFeaturePipeline()
    bars = _bars(n=200)
    with pytest.raises(ValueError):
        pipe.fit(bars[: pipe.warmup - 1])


def test_transform_short_window_raises():
    pipe = RollingFeaturePipeline()
    bars = _bars()
    pipe.fit(bars)
    with pytest.raises(ValueError):
        pipe.transform(bars[-(pipe.warmup - 1) :])


def test_no_nan_or_inf_in_output():
    bars = _bars()
    pipe = RollingFeaturePipeline()
    pipe.fit(bars)
    # Sweep many distinct windows across the series, not just the tail.
    for end in range(pipe.warmup, len(bars), 37):
        fv = pipe.transform(bars[end - pipe.warmup : end])
        arr = np.asarray(fv.values, dtype=float)
        assert arr.shape == (len(pipe.feature_names),)
        assert all(math.isfinite(v) for v in fv.values)


def test_transform_deterministic_for_same_input():
    bars = _bars()
    pipe = RollingFeaturePipeline()
    pipe.fit(bars)
    window = bars[-pipe.warmup :]
    a = pipe.transform(window)
    b = pipe.transform(window)
    assert a.values == b.values


def test_scaler_is_fit_only_not_refit_on_transform():
    """Transforming the same window twice (and an unrelated one in between) must
    not move the scaler — output for the original window stays identical."""
    bars = _bars()
    pipe = RollingFeaturePipeline()
    pipe.fit(bars)
    window = bars[-pipe.warmup :]
    first = pipe.transform(window).values
    # Interleave a very different window to expose any accidental partial_fit.
    pipe.transform(bars[pipe.warmup : 2 * pipe.warmup])
    second = pipe.transform(window).values
    assert first == second
