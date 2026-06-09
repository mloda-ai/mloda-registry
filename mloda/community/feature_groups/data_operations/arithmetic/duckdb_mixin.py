"""Shared DuckDB classmethods for the point- and scalar-arithmetic families.

The DuckDB point and scalar arithmetic backends carry byte-for-byte-identical
``compute_framework_rule`` / ``_input_columns_and_framework`` /
``_assert_source_column_is_numeric`` implementations, plus the same
``DUCKDB_ARITHMETIC_OPS`` operator map. This mixin holds the one copy; the
concrete DuckDB backends inherit it (first in their MRO), import
``DUCKDB_ARITHMETIC_OPS`` from here, and supply only ``_compute_arithmetic``.
Keeping one mixin per module preserves optional-dependency isolation: this
module imports only the DuckDB backend.
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.arithmetic.base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.arithmetic.duckdb_numeric_source import (
    duckdb_non_numeric_descriptor,
)

DUCKDB_ARITHMETIC_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}


class DuckdbArithmeticMixin(ArithmeticFeatureGroupBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: DuckdbRelation) -> tuple[list[str], str]:
        return list(data.columns), "DuckDB"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: DuckdbRelation, source_col: str) -> None:
        descriptor = duckdb_non_numeric_descriptor(data, source_col)
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)
