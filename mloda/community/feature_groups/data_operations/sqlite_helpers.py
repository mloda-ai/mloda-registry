"""Shared SQLite helper utilities: declared column affinities and a native storage-class probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

if TYPE_CHECKING:
    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


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


def wall_clock_expr(quoted: str) -> str:
    """SQL expression that strips a trailing ``+HH:MM`` / ``-HH:MM`` / ``Z``, leaving the wall clock."""
    cond = _HAS_OFFSET.format(q=quoted)
    return (
        f"(CASE WHEN {cond} THEN substr({quoted}, 1, length({quoted}) - 6) "
        f"WHEN substr({quoted}, -1) = 'Z' THEN substr({quoted}, 1, length({quoted}) - 1) "
        f"ELSE {quoted} END)"
    )


def offset_suffix_expr(quoted: str) -> str:
    """SQL expression for the trailing offset or ``''``, normalizing UTC ``Z`` to ``+00:00``."""
    cond = _HAS_OFFSET.format(q=quoted)
    return f"(CASE WHEN {cond} THEN substr({quoted}, -6) WHEN substr({quoted}, -1) = 'Z' THEN '+00:00' ELSE '' END)"
