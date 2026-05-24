"""Info-theoretic strategy: trade the CHANGE in regime-assignment entropy.

Computes Shannon binary entropy H of regime_prob at each bar, tracks a
per-symbol EMA of H, and derives the gradient dH = H - ema_H.

Falling entropy (dH < 0) means the regime is crystallising — information is
being *gained* — so we scale UP the drift position.  Rising entropy means the
regime is dissolving, so we de-risk.

target_weight = tanh(gain * tanh(sharpe))
              * clamp(0.2 - k * dH, 0, 1)
              * (1 - anomaly_score)

Flat on ANOMALY_REGIME or anomaly_score > ANOMALY_THRESH.
Dead-band: |weight| < DEADBAND -> 0.
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

NAME = "entropy_gradient"
DESCRIPTION = (
    "Information-gradient strategy: takes positions when Shannon entropy of "
    "regime_prob is FALLING (regime crystallising) and de-risks when entropy "
    "rises.  target_weight = tanh(gain*tanh(sharpe)) * clamp(0.2-k*dH,0,1) "
    "* (1-anomaly_score).  Per-symbol EMA tracks entropy level; no lookahead."
)

# --- tunables --------------------------------------------------------------- #
_EPS = 1e-9
_P_CLAMP = 1e-6          # keep log2 arguments away from 0
_GAIN = 60.0             # tanh amplifier on the Sharpe proxy (matches entropy_sizing)
_EMA_ALPHA = 0.1         # EMA smoothing for H; ~10-bar half-life
_K = 2.0                 # sensitivity of scaling factor to entropy gradient
_BASE = 0.2              # target scale when dH == 0
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> hard flat
_DEADBAND = 0.04         # collapse |weight| below this to zero


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits, p clamped to [_P_CLAMP, 1-_P_CLAMP]."""
    p = max(_P_CLAMP, min(1.0 - _P_CLAMP, p))
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


class EntropyGradientStrategy:
    """Trade the sign and magnitude of the entropy GRADIENT, not entropy itself.

    State per symbol: ema_H — exponential moving average of H(regime_prob).
    """

    def __init__(
        self,
        gain: float = _GAIN,
        ema_alpha: float = _EMA_ALPHA,
        k: float = _K,
        base: float = _BASE,
        anomaly_thresh: float = _ANOMALY_THRESH,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._alpha = ema_alpha
        self._k = k
        self._base = base
        self._anomaly_thresh = anomaly_thresh
        self._deadband = deadband
        self._ema_H: dict[str, float] = {}   # per-symbol EMA of H

    # ---------------------------------------------------------------------- #
    #  signals()                                                              #
    # ---------------------------------------------------------------------- #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        H = _binary_entropy(state.regime_prob)

        # Update per-symbol EMA; initialise on first encounter.
        ema = self._ema_H.get(sym, H)
        ema = self._alpha * H + (1.0 - self._alpha) * ema
        self._ema_H[sym] = ema

        dH = H - ema   # negative => entropy falling => regime crystallising

        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # Pack dH into the confidence slot; momentum carries the Sharpe proxy.
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=dH,           # reused as entropy gradient carrier
        )

    # ---------------------------------------------------------------------- #
    #  target()                                                               #
    # ---------------------------------------------------------------------- #
    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates.
        if signals.regime_id == ANOMALY_REGIME:
            return flat
        if signals.anomaly > self._anomaly_thresh:
            return flat

        dH = signals.confidence   # entropy gradient stored here
        mom = max(-1.0 + _EPS, min(1.0 - _EPS, signals.momentum))

        # Entropy-gradient scale factor: clamp to [0, 1].
        scale = max(0.0, min(1.0, self._base - self._k * dH))

        weight = (
            math.tanh(self._gain * math.atanh(mom))
            * scale
            * max(0.0, 1.0 - signals.anomaly)
        )

        # Dead-band.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh EntropyGradientStrategy instance."""
    return EntropyGradientStrategy()
