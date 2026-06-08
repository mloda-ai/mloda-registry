"""DuckDB implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from typing import Any

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.duckdb_arithmetic_mixin import (
    DUCKDB_ARITHMETIC_OPS as DUCKDB_ARITHMETIC_OPS,  # re-exported for backend-parity tests
    DuckdbArithmeticMixin,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)


class DuckdbPointArithmetic(DuckdbArithmeticMixin, PointArithmeticFeatureGroup):
    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> DuckdbRelation:
        sql_op = DUCKDB_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise unsupported_op_error(op, DUCKDB_ARITHMETIC_OPS, framework="DuckDB")

        quoted_a = quote_ident(col_a)
        quoted_b = quote_ident(col_b)
        quoted_feature = quote_ident(feature_name)

        # Cast both operands for divide to guarantee float division and
        # IEEE-754 inf/nan semantics; native SQL int/int would truncate.
        if op == "divide":
            left = f"CAST({quoted_a} AS DOUBLE)"
            right = f"CAST({quoted_b} AS DOUBLE)"
        else:
            left = quoted_a
            right = quoted_b

        raw_sql = f"*, ({left} {sql_op} {right}) AS {quoted_feature}"
        result: DuckdbRelation = data.project(raw_sql)
        return result
