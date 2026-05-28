"""DuckDB implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)

DUCKDB_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}

# DuckDB type names that count as numeric for point arithmetic.
# Parameterized variants (DECIMAL(p, s)) are matched via ``startswith(p + "(")``.
_DUCKDB_NUMERIC_PREFIXES: tuple[str, ...] = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
    "NUMERIC",
    "BIGNUM",
)


class DuckdbPointArithmetic(PointArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: DuckdbRelation) -> tuple[list[str], str]:
        return list(data.columns), "DuckDB"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: DuckdbRelation, source_col: str) -> None:
        # ``DuckdbRelation`` wraps a ``DuckDBPyRelation`` exposing aligned
        # ``.columns`` and ``.types`` (~4 microseconds; cheaper than
        # ``data.to_arrow_table().schema`` which materializes the relation).
        underlying = data._relation
        type_by_column = dict(zip(list(underlying.columns), [str(t) for t in underlying.types]))
        dtype_str = type_by_column.get(source_col)
        if dtype_str is None:
            return
        if not any(dtype_str == p or dtype_str.startswith(p + "(") for p in _DUCKDB_NUMERIC_PREFIXES):
            cls._raise_non_numeric_source(source_col, dtype_str)

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
