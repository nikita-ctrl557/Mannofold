"""KL-Divergence Regime Strategy.

Information-theoretic regime detector: computes KL(p||q) where p is the
current bar's Gaussian (fwd_return_mean, fwd_return_std) and q is a per-symbol
running Gaussian maintained via EMA of mean and variance.

Large KL  => state has diverged from its recent norm => regime shift signal.
Small KL  => nothing new => small position.

target_weight = sign(fwd_return_mean) * tanh(gain * clamp(KL, 0, 5)) * confidence
  where confidence = regime_prob * (1 - anomaly_score).

Flat on ANOMALY_REGIME or anomaly_score > 0.6.
Dead-band |w| < 0.04 -> 0.
"""

from __future__ import annotations

import math
from typing import Dict

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "kl_divergence_regime"
DESCRIPTION = (
    "Information-theoretic strategy: computes KL divergence between the current "
    "bar's Gaussian and a running (EMA) reference Gaussian per symbol. Large KL "
    "signals a regime shift worth trading; weight scales as tanh(gain*KL) * confidence."
)

_EPS            = 1e-9
_GAIN           = 2.0
_ANOMALY_THRESH = 0.6
_DEAD_BAND      = 0.04
_KL_MAX         = 5.0

# EMA decay for the running reference Gaussian
_MU_ALPHA  = 0.05   # slow mean tracker
_VAR_ALPHA = 0.05   # slow variance tracker
_VAR_INIT  = 1e-4   # initial reference variance (small but non-zero)


class _SymbolState:
    """Running Gaussian q(mu_q, sigma_q) updated via EMA."""

    def __init__(self, mu0: float, var0: float) -> None:
        self.mu  = mu0
        self.var = var0

    def kl_and_update(self, mu_p: float, var_p: float) -> float:
        """Return KL(p||q) then update the running reference."""
        mu_q  = self.mu
        var_q = max(self.var, _EPS)
        var_p = max(var_p, _EPS)

        # KL divergence between two Gaussians:
        # KL(p||q) = log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2*sigma_q^2) - 0.5
        kl = (
            0.5 * math.log(var_q / var_p)
            + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
            - 0.5
        )
        kl = max(0.0, kl)   # numerical guard (should be >= 0 by definition)

        # Update running reference via EMA AFTER computing KL (no lookahead)
        self.mu  = _MU_ALPHA  * mu_p  + (1.0 - _MU_ALPHA)  * self.mu
        self.var = _VAR_ALPHA * var_p + (1.0 - _VAR_ALPHA) * self.var

        return kl


class KLDivergenceRegime:
    """Per-symbol KL-divergence regime detector."""

    def __init__(self) -> None:
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str, mu0: float, var0: float) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(mu0, var0)
        return self._states[symbol]

    def signals(self, state: ManifoldState) -> SignalSet:
        mu_p  = state.fwd_return_mean
        std_p = state.fwd_return_std
        var_p = std_p * std_p

        confidence = max(0.0, min(1.0,
            state.regime_prob * (1.0 - state.anomaly_score)
        ))

        # Flat when anomalous
        if state.regime_id == ANOMALY_REGIME or state.anomaly_score > _ANOMALY_THRESH:
            return SignalSet(
                ts=state.ts,
                symbol=state.symbol,
                momentum=0.0,
                expected_return=mu_p,
                anomaly=state.anomaly_score,
                regime_id=state.regime_id,
                confidence=0.0,
            )

        sym_state = self._get_state(state.symbol, mu_p, max(var_p, _VAR_INIT))
        kl = sym_state.kl_and_update(mu_p, var_p)

        # Direction follows the drift; magnitude scales with KL surprise
        drift_sign = 1.0 if mu_p >= 0.0 else -1.0
        clamped_kl = min(kl, _KL_MAX)
        momentum = drift_sign * math.tanh(_GAIN * clamped_kl)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=mu_p,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        raw = signals.momentum * signals.confidence

        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh KLDivergenceRegime instance."""
    return KLDivergenceRegime()
