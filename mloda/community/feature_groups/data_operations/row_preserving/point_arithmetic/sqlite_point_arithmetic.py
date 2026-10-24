"""SQLite implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import SQL_ARITHMETIC_OPS
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.sqlite_mixin import (
    SqliteArithmeticMixin,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)


class SqlitePointArithmetic(Implementation[PointArithmeticFeatureGroup]):
    feature_group = PointArithmeticFeatureGroup
    compute_framework = SqliteFramework
    divergent = {"divide": "returns NULL on divide by zero"}

    @classmethod
    def calculate(
        cls,
        group: type[PointArithmeticFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.apply_arithmetic(
            data,
            features,
            non_numeric=SqliteArithmeticMixin._non_numeric_descriptor,
            compute_arithmetic=cls._compute_arithmetic,
        )

    @classmethod
    def _compute_arithmetic(
        cls,
        data: SqliteRelation,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> SqliteRelation:
        sql_op = SQL_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise unsupported_op_error(op, SQL_ARITHMETIC_OPS, framework="SQLite")

        quoted_a = quote_ident(col_a)
        quoted_b = quote_ident(col_b)
        quoted_feature = quote_ident(feature_name)

        # Cast both operands for divide to guarantee float division;
        # native SQL int/int would truncate. SQLite returns NULL on
        # divide-by-zero (no IEEE-754 storage), which is the documented
        # divergence from PyArrow/Pandas/Polars/DuckDB.
        if op == "divide":
            left = f"CAST({quoted_a} AS REAL)"
            right = f"CAST({quoted_b} AS REAL)"
        else:
            left = quoted_a
            right = quoted_b

        # Row-preserving contract: append_column below aligns result_values to
        # the existing rows by position, so this bare SELECT must return rows in
        # the relation's stored order. Keep it free of ORDER BY / JOIN / GROUP BY
        # / DISTINCT, any of which could reorder rows and silently misalign the
        # appended column. See docs/guides/data-operation-patterns/02-row-preserving-contract.md.
        sql = " ".join(
            [
                "SELECT",
                f"({left} {sql_op} {right}) AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
