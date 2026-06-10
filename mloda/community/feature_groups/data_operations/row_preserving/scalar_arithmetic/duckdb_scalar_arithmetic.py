"""DuckDB implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.duckdb_mixin import (
    DUCKDB_ARITHMETIC_OPS,
    DuckdbArithmeticMixin,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)


class DuckdbScalarArithmetic(DuckdbArithmeticMixin, ScalarArithmeticFeatureGroup):
    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> DuckdbRelation:
        sql_op = DUCKDB_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise unsupported_op_error(op, DUCKDB_ARITHMETIC_OPS, framework="DuckDB")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        # Preserve int-vs-float in SQL literal so int + int stays int.
        literal = repr(constant) if isinstance(constant, int) else repr(float(constant))

        # Cast only for divide, to avoid SQL integer-division truncation
        # (e.g. 10 / 3 -> 3). For add/subtract/multiply, native arithmetic
        # preserves the source dtype and matches the columnar backends.
        if op == "divide":
            source_expr = f"CAST({quoted_source} AS DOUBLE)"
        else:
            source_expr = quoted_source

        raw_sql = f"*, ({source_expr} {sql_op} {literal}) AS {quoted_feature}"
        result: DuckdbRelation = data.project(raw_sql)
        return result
