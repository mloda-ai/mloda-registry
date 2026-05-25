"""DuckDB implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)

DUCKDB_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}

# DuckDB type names that count as numeric for scalar arithmetic.
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


class DuckdbScalarArithmetic(ScalarArithmeticFeatureGroup):
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
