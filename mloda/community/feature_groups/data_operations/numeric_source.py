"""Single source of truth for "what counts as numeric" per compute framework.

The point-arithmetic and scalar-arithmetic families each reject non-numeric
source columns up-front (CFW backends raise a clear ``ValueError`` rather than
silently emulating in Python). Before this module existed, every backend in
both families carried its own copy of the per-framework numeric-type check.

That duplication is the bug behind issue #214: a numeric type added to one
family's allowlist and forgotten in the other would silently reject valid
columns on one operation only (e.g. ``BIGNUM`` accepted by point arithmetic
but rejected by scalar arithmetic). Centralising the introspection here means
both families share one definition of numeric-ness per backend.

Each helper returns a *descriptor* of the rejected type when the column is
NON-numeric (so the caller can inline it into the shared error message via
``ArithmeticFeatureGroupBase._raise_non_numeric_source``), and ``None`` when
the column IS numeric or is absent (absence is handled separately by the
calculate_feature column-presence check).

pandas / polars / pyarrow are optional dependencies that differ per family, so
they are imported lazily inside each helper rather than at module top: this
module is imported by both families regardless of which frameworks are
installed, so a hard import would break environments that only install one.
"""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

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


def sqlite_non_numeric_descriptor(data: Any, source_col: str) -> str | None:
    """Return ``f"SQLite affinity {affinity!r}"`` when NON-numeric, else ``None``.

    Uses ``PRAGMA table_info`` declared affinity. Returns ``None`` when the
    column is absent (presence is validated separately by the caller).

    Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
    ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``), so a
    boolean source column is indistinguishable from ``int64`` at the relation
    level. The shared boolean-source tests are correspondingly skipped for
    SQLite via the ``detects_non_numeric_source`` test-class override.
    """
    rows = data.connection.execute(f"PRAGMA table_info({quote_ident(data.table_name)})").fetchall()
    affinity_by_column = {row[1]: (row[2] or "").upper() for row in rows}
    affinity = affinity_by_column.get(source_col)
    if affinity is None:
        return None
    if "INT" in affinity or "REAL" in affinity or "FLOA" in affinity or "DOUB" in affinity or "NUMERIC" in affinity:
        return None
    return f"SQLite affinity {affinity!r}"


def pandas_non_numeric_descriptor(series: Any) -> Any | None:
    """Return ``series.dtype`` when the pandas series is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    import pandas as pd

    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return series.dtype
    return None


def polars_non_numeric_descriptor(dtype: Any) -> Any | None:
    """Return ``dtype`` when the polars dtype is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    import polars as pl

    if dtype == pl.Boolean or not dtype.is_numeric():
        return dtype
    return None


def pyarrow_non_numeric_descriptor(arrow_type: Any) -> Any | None:
    """Return ``arrow_type`` when the pyarrow type is NON-numeric, else ``None``.

    Booleans count as NON-numeric.
    """
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type) or not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
        return arrow_type
    return None
