"""Shared SQLite helper utilities: declared column affinities and a native storage-class probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

if TYPE_CHECKING:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


def sqlite_numeric_type(arrow_type: pa.DataType, *, real: bool = False) -> str | None:
    """``INTEGER`` for an integer arrow type, ``REAL`` for a float (or when ``real``), else ``None``."""
    if real:
        return "REAL"
    if pa.types.is_integer(arrow_type):
        return "INTEGER"
    if pa.types.is_floating(arrow_type):
        return "REAL"
    return None


def column_affinities(data: "SqliteRelation") -> dict[str, str]:
    """Column name -> upper-cased declared type from ``PRAGMA table_info``; '' marks a computed column."""
    rows = data.connection.execute(f"PRAGMA table_info({quote_ident(data.table_name)})").fetchall()
    return {row[1]: (row[2] or "").upper() for row in rows}


def first_storage_class_outside(data: "SqliteRelation", column: str, allowed: frozenset[str]) -> str | None:
    """First non-null value's ``typeof`` outside ``allowed`` (lower-case classes), or ``None``."""
    ordered = sorted(allowed)
    placeholders = ", ".join("?" for _ in ordered)
    sql = (
        f"SELECT typeof({quote_ident(column)}) "  # nosec
        f"FROM {quote_ident(data.table_name)} "
        f"WHERE {quote_ident(column)} IS NOT NULL AND typeof({quote_ident(column)}) NOT IN ({placeholders}) "
        f"LIMIT 1"
    )
    row = data.connection.execute(sql, ordered).fetchone()
    return None if row is None else str(row[0])


_HAS_OFFSET = "substr({q}, -6, 1) IN ('+', '-') AND substr({q}, -3, 1) = ':'"
_HAS_Z = "upper(substr({q}, -1)) = 'Z'"


def wall_clock_expr(quoted: str) -> str:
    """SQL expression that strips a trailing ``+HH:MM`` / ``-HH:MM`` offset or ``Z``, leaving the wall clock."""
    cond = _HAS_OFFSET.format(q=quoted)
    return (
        f"(CASE WHEN {cond} THEN substr({quoted}, 1, length({quoted}) - 6) "
        f"WHEN {_HAS_Z.format(q=quoted)} THEN substr({quoted}, 1, length({quoted}) - 1) ELSE {quoted} END)"
    )


def offset_suffix_expr(quoted: str) -> str:
    """SQL expression that yields the trailing ``+HH:MM`` / ``-HH:MM`` offset (``Z`` as ``+00:00``) or ``''``."""
    cond = _HAS_OFFSET.format(q=quoted)
    return f"(CASE WHEN {cond} THEN substr({quoted}, -6) WHEN {_HAS_Z.format(q=quoted)} THEN '+00:00' ELSE '' END)"
