"""SQLite implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)

# SQLite's native UPPER/LOWER are ASCII-only: UPPER('héllo') returns 'HéLLO'
# instead of 'HÉLLO'. Rather than emulate unicode-aware semantics in Python
# and risk silent divergence from the PyArrow reference, SQLite refuses to
# serve upper/lower and declares only the ops below as supported, so the
# resolver falls back to another framework. The same holds for 'reverse',
# which SQLite has no native function for.
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "trim": "TRIM({col})",
    "length": "LENGTH({col})",
}


class SqliteStringOps(Implementation[StringFeatureGroup]):
    feature_group = StringFeatureGroup
    compute_framework = SqliteFramework
    supported = frozenset(_SQLITE_STRING_EXPRS)

    @classmethod
    def calculate(
        cls, group: type[StringFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.transform_string(data, features, compute_string=cls._compute_string)

    @classmethod
    def _compute_string(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> SqliteRelation:
        expr_template = _SQLITE_STRING_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported string operation for SQLite: {op}")

        quoted_source = quote_ident(source_col)
        expr = expr_template.format(col=quoted_source)
        quoted_feature = quote_ident(feature_name)

        sql = " ".join(
            [
                "SELECT",
                f"{expr} AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY rowid",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
