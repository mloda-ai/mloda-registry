"""Collision-free names for internal helper columns in the non-SQL data-operations backends.

Several row-preserving and aggregation implementations add temporary helper
columns (for example a row-index to restore input order after a reordering
window function) and drop them again at the end. Rather than banning a
``__mloda_`` prefix on user input, these backends pick a helper name that is
provably absent from the current frame.

Call ``unique_helper_name(base, taken)`` with the set of names already present
(input columns, the output feature name, and any helper picked earlier in the
same operation) as ``taken``. ``data.columns`` (pandas), a plain ``set``, and
``table.column_names`` (pyarrow) all support ``in``, so ``Container[str]`` is
the right type for ``taken``.

The SQL backends (DuckDB, SQLite) use their own ``pick_helper_column_name`` in
the SQL utils and are intentionally not covered here.
"""

from __future__ import annotations

from collections.abc import Container


def unique_helper_name(base: str, taken: Container[str]) -> str:
    """Return ``base`` if absent from ``taken``, else the lowest ``base_N`` (N>=1) not in ``taken``."""
    if base not in taken:
        return base
    i = 1
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"
