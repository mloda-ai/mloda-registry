"""DuckDB "what counts as numeric" source-column check for arithmetic.

Shared by the point-arithmetic and scalar-arithmetic families so both reject
the same set of non-numeric DuckDB source columns up-front.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.duckdb_helpers import column_types

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
    type_by_column = column_types(data)
    dtype_str: str | None = type_by_column.get(source_col)
    if dtype_str is None:
        return None
    if not any(dtype_str == p or dtype_str.startswith(p + "(") for p in DUCKDB_NUMERIC_PREFIXES):
        return dtype_str
    return None
