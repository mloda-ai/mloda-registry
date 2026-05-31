"""DuckDB "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric DuckDB source columns up-front.
"""

from __future__ import annotations

from typing import Any

# DuckDB type names that count as numeric for arithmetic.
# Parameterized variants (DECIMAL(p, s)) are matched via ``startswith(p + "(")``.
DUCKDB_NUMERIC_PREFIXES: tuple[str, ...] = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
    "NUMERIC",
    "BIGNUM",
)


def duckdb_non_numeric_descriptor(data: Any, source_col: str) -> str | None:
    """Return the DuckDB dtype string when ``source_col`` is NON-numeric, else ``None``.

    Returns ``None`` when the column is absent (presence is validated
    separately by the calling feature group).
    """
    # ``DuckdbRelation`` wraps a ``DuckDBPyRelation`` exposing aligned
    # ``.columns`` and ``.types`` (~4 microseconds; cheaper than
    # ``data.to_arrow_table().schema`` which materializes the relation).
    underlying = data._relation
    type_by_column = dict(zip(list(underlying.columns), [str(t) for t in underlying.types]))
    dtype_str: str | None = type_by_column.get(source_col)
    if dtype_str is None:
        return None
    if not any(dtype_str == p or dtype_str.startswith(p + "(") for p in DUCKDB_NUMERIC_PREFIXES):
        return dtype_str
    return None
