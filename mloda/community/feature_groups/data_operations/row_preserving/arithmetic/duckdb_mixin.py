"""Shared DuckDB classmethods for the point- and scalar-arithmetic families.

``DuckdbArithmeticMixin`` is a plain runtime class typed against
``ArithmeticFeatureGroupBase`` for mypy only, so it stays out of FeatureGroup
plugin discovery. It supplies ``compute_framework_rule``,
``_input_columns_and_framework``, and the ``_non_numeric_descriptor`` hook
consumed by the base's ``_assert_source_column_is_numeric`` template; the
structural guards live in ``tests/test_numeric_source.py``. The concrete
DuckDB backends import ``DUCKDB_ARITHMETIC_OPS`` from here (an alias of the
shared ``SQL_ARITHMETIC_OPS``). Keeping one mixin per module preserves
optional-dependency isolation: this module imports only the DuckDB backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import SQL_ARITHMETIC_OPS
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.duckdb_numeric_source import (
    duckdb_non_numeric_descriptor,
)

if TYPE_CHECKING:
    from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.base import ArithmeticFeatureGroupBase

    _ArithmeticMixinBase = ArithmeticFeatureGroupBase
else:
    _ArithmeticMixinBase = object

DUCKDB_ARITHMETIC_OPS: dict[str, str] = SQL_ARITHMETIC_OPS


class DuckdbArithmeticMixin(_ArithmeticMixinBase):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: DuckdbRelation) -> tuple[list[str], str]:
        return list(data.columns), "DuckDB"

    @classmethod
    def _non_numeric_descriptor(cls, data: DuckdbRelation, source_col: str) -> object | None:
        return duckdb_non_numeric_descriptor(data, source_col)
