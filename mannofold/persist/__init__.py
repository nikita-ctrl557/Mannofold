"""Persistence: partitioned Parquet for bars/steps + DuckDB query surface.

``LocalStateStore`` is the reference implementation. A Supabase exporter (added by
the persistence workstream) implements the same Protocol behind the scenes and is
never on the hot path.
"""

from mannofold.persist.store import LocalStateStore

__all__ = ["LocalStateStore"]
