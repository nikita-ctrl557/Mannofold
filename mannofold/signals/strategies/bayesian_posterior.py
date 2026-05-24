"""Bayesian posterior strategy.

Estimates P(up) via Bayes' theorem:
  - PRIOR from regime: prior_up = 0.5 + 0.5 * tanh(k * regime_drift_sign * regime_prob)
  - LIKELIHOOD from z-score of fwd_return_mean mapped via logistic to evidence ratio
  - POSTERIOR odds = prior_odds * likelihood_ratio
  - Position sized by posterior EDGE over 50%, scaled by confidence
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "bayesian_posterior"
DESCRIPTION = (
    "Bayesian posterior strategy: estimates P(up) from prior (regime) and "
    "likelihood (z-score evidence), sizes position by posterior edge over 50%."
)

_EPS             = 1e-9
_PRIOR_K         = 2.0    # steepness of regime prior tanh
_LOGISTIC_SCALE  = 1.0    # scale for z-score -> likelihood logistic
_GAIN            = 2.5    # tanh gain on 2*(p-0.5) edge
_ANOMALY_THRESH  = 0.6    # hard anomaly cutoff
_DEAD_BAND       = 0.04   # minimum |weight| to trade
_EDGE_THRESH     = 0.02   # minimum |p-0.5| to trade


def _logistic(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def _compute_prior(regime_id: int, regime_prob: float, fwd_return_mean: float) -> float:
    """Compute prior P(up) from regime information."""
    if regime_id == ANOMALY_REGIME:
        return 0.5
    # regime drift sign from expected return direction
    drift_sign = 1.0 if fwd_return_mean > 0.0 else (-1.0 if fwd_return_mean < 0.0 else 0.0)
    return 0.5 + 0.5 * math.tanh(_PRIOR_K * drift_sign * regime_prob)


def _compute_posterior(prior_up: float, z: float) -> float:
    """Compute posterior P(up) using Bayes with logistic likelihood."""
    # prior odds
    prior_odds = prior_up / (1.0 - prior_up + _EPS)

    # likelihood ratio: logistic(z) / logistic(-z) = exp(z) (via logistic symmetry)
    # P(evidence | up) = logistic(z), P(evidence | down) = logistic(-z)
    likelihood_ratio = math.exp(max(-20.0, min(20.0, z)))

    # posterior odds and probability
    post_odds = prior_odds * likelihood_ratio
    return post_odds / (1.0 + post_odds)


class BayesianPosteriorStrategy:
    """Trade based on Bayesian posterior P(up), sized by posterior edge."""

    def signals(self, state: ManifoldState) -> SignalSet:
        z = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=z,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard-off conditions
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return flat

        # Retrieve stored state info from signals
        # Prior from regime + direction of expected return (embedded in expected_return)
        # We need regime info - approximate regime drift sign from expected_return sign
        drift_sign = 1.0 if signals.expected_return > 0.0 else (-1.0 if signals.expected_return < 0.0 else 0.0)
        prior_up = 0.5 + 0.5 * math.tanh(_PRIOR_K * drift_sign * signals.confidence)

        # z-score from signals.momentum
        z = signals.momentum

        # Posterior P(up)
        p = _compute_posterior(prior_up, _LOGISTIC_SCALE * z)

        # Flat if edge below threshold
        edge = p - 0.5
        if abs(edge) < _EDGE_THRESH:
            return flat

        # Target weight = tanh(gain * 2 * (p - 0.5)) * confidence
        raw = math.tanh(_GAIN * 2.0 * edge) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            return flat

        raw = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh BayesianPosteriorStrategy instance."""
    return BayesianPosteriorStrategy()
