"""SQLite implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any, Callable

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)

# SQLite native string functions whose behavior matches the PyArrow reference
# for the canonical test data (TRIM strips ASCII whitespace; LENGTH on TEXT
# returns codepoints, same as Python len()).
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "trim": "TRIM({col})",
    "length": "LENGTH({col})",
}

# Ops that SQLite CAN handle in SQL but ASCII-only, so we apply Python's
# Unicode-aware equivalents post-fetch instead. Matches PyArrow reference.
_PYTHON_STRING_FUNCS: dict[str, Callable[[str], Any]] = {
    "upper": str.upper,
    "lower": str.lower,
}

# reverse is not supported natively in SQLite.
_SUPPORTED_OPS: frozenset[str] = frozenset(_SQLITE_STRING_EXPRS) | frozenset(_PYTHON_STRING_FUNCS)


class SqliteStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Reject 'reverse' at match time since SQLite has no native reverse function."""
        return operation_config in _SUPPORTED_OPS

    @classmethod
    def _compute_string(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> SqliteRelation:

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        if op in _PYTHON_STRING_FUNCS:
            func = _PYTHON_STRING_FUNCS[op]
            sql = " ".join(
                [
                    "SELECT",
                    f"{quoted_source} AS {quoted_feature},",
                    f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                    "FROM",
                    f"{quote_ident(data.table_name)}",
                    "ORDER BY",
                    qrn,
                ]
            )
            cursor = data.connection.execute(sql)
            rows = cursor.fetchall()
            result_values = [None if row[0] is None else func(row[0]) for row in rows]
            return data.append_column(feature_name, result_values)

        expr_template = _SQLITE_STRING_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported string operation for SQLite: {op}")

        expr = expr_template.format(col=quoted_source)

        sql = " ".join(
            [
                "SELECT",
                f"{expr} AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
