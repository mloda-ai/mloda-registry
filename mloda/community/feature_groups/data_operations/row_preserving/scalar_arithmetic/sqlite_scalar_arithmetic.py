"""SQLite implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.arithmetic.sqlite_mixin import (
    SQLITE_ARITHMETIC_OPS,
    SqliteArithmeticMixin,
)


class SqliteScalarArithmetic(SqliteArithmeticMixin, ScalarArithmeticFeatureGroup):
    @classmethod
    def _compute_arithmetic(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> SqliteRelation:
        sql_op = SQLITE_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise unsupported_op_error(op, SQLITE_ARITHMETIC_OPS, framework="SQLite")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        # Preserve int-vs-float in SQL literal so int + int stays int.
        literal = repr(constant) if isinstance(constant, int) else repr(float(constant))

        # Cast only for divide, to avoid SQL integer-division truncation
        # (e.g. 10 / 3 -> 3). For add/subtract/multiply, native arithmetic
        # preserves the source dtype and matches the columnar backends.
        if op == "divide":
            source_expr = f"CAST({quoted_source} AS REAL)"
        else:
            source_expr = quoted_source

        # Row-preserving contract: append_column below aligns result_values to
        # the existing rows by position, so this bare SELECT must return rows in
        # the relation's stored order. Keep it free of ORDER BY / JOIN / GROUP BY
        # / DISTINCT, any of which could reorder rows and silently misalign the
        # appended column. See docs/guides/data-operation-patterns/02-row-preserving-contract.md.
        sql = " ".join(
            [
                "SELECT",
                f"({source_expr} {sql_op} {literal}) AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)
