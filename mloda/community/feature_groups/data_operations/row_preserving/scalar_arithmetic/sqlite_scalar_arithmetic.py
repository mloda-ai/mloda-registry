"""SQLite implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

_SQLITE_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}


class SqliteScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _compute_arithmetic(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> SqliteRelation:
        sql_op = _SQLITE_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise ValueError(
                f"Unsupported arithmetic operation for SQLite: {op!r}. Supported: {sorted(_SQLITE_ARITHMETIC_OPS)}."
            )

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        literal = float(constant)

        sql = " ".join(
            [
                "SELECT",
                f"(CAST({quoted_source} AS REAL) {sql_op} {literal}) AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
