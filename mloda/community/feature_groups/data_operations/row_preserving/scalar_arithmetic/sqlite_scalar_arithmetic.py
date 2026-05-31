"""SQLite implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.sqlite_numeric_source import sqlite_non_numeric_descriptor
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

SQLITE_ARITHMETIC_OPS: dict[str, str] = {
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
    def _input_columns_and_framework(cls, data: SqliteRelation) -> tuple[list[str], str]:
        return list(data.columns), "SQLite"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: SqliteRelation, source_col: str) -> None:
        """Reject non-numeric source columns via ``PRAGMA table_info`` declared affinity.

        Caveat: ``SqliteRelation.from_arrow`` maps arrow booleans to SQLite
        ``INTEGER`` affinity (see ``mloda_plugins`` ``_arrow_type_to_sqlite``),
        so a boolean source column is indistinguishable from ``int64`` at the
        relation level. The shared test ``test_boolean_source_column_rejected``
        is correspondingly skipped for SQLite via the
        ``detects_non_numeric_source`` test-class override.
        """
        descriptor = sqlite_non_numeric_descriptor(data, source_col)
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)

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
