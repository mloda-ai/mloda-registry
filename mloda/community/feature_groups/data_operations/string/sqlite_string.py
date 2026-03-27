"""SQLite implementation for string operation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)

# SQLite native string functions.
# reverse is not supported natively in SQLite.
_SQLITE_STRING_EXPRS: dict[str, str] = {
    "upper": "UPPER({col})",
    "lower": "LOWER({col})",
    "trim": "TRIM({col})",
    "length": "LENGTH({col})",
}


class SqliteStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Reject 'reverse' at match time since SQLite has no native reverse function."""
        return operation_config in _SQLITE_STRING_EXPRS

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
        qrn = quote_ident("__mloda_rn__")

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
