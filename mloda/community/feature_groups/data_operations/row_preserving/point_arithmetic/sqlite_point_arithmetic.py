"""SQLite implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.sqlite_arithmetic_mixin import (
    SQLITE_ARITHMETIC_OPS as SQLITE_ARITHMETIC_OPS,  # re-exported for backend-parity tests
    SqliteArithmeticMixin,
)


class SqlitePointArithmetic(SqliteArithmeticMixin, PointArithmeticFeatureGroup):
    @classmethod
    def _compute_arithmetic(
        cls,
        data: SqliteRelation,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> SqliteRelation:
        sql_op = SQLITE_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise unsupported_op_error(op, SQLITE_ARITHMETIC_OPS, framework="SQLite")

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
