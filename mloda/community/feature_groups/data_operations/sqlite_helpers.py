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
