"""DuckDB implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

_DUCKDB_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}


class DuckdbScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _compute_arithmetic(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> DuckdbRelation:
        sql_op = _DUCKDB_ARITHMETIC_OPS.get(op)
        if sql_op is None:
            raise ValueError(
                f"Unsupported arithmetic operation for DuckDB: {op!r}. Supported: {sorted(_DUCKDB_ARITHMETIC_OPS)}."
            )

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        literal = float(constant)

        raw_sql = f"*, (CAST({quoted_source} AS DOUBLE) {sql_op} {literal}) AS {quoted_feature}"
        result: DuckdbRelation = data.project(raw_sql)
        return result
