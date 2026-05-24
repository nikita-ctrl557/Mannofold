"""Generate a synthetic regime-switching dataset and persist it as Parquet.

Usage: python scripts/gen_synthetic.py [n_bars] [seed]
"""

from __future__ import annotations

import sys

from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    bars, labels = generate_bars(SyntheticConfig(n_bars=n, seed=seed))
    LocalStateStore().append_bars(bars)
    counts = {s: labels.count(s) for s in set(labels)}
    print(f"generated {len(bars)} bars  |  regime counts: {counts}")
    print(f"price range: {min(b.close for b in bars):.2f} .. {max(b.close for b in bars):.2f}")


if __name__ == "__main__":
    main()
