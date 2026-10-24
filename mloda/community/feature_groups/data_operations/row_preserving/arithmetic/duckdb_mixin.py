"""Shared duckdb classmethods for the point- and scalar-arithmetic families.

``DuckdbArithmeticMixin`` is a plain class supplying the framework's
``_non_numeric_descriptor``, which the point and scalar Implementations pass to
the group's ``apply_arithmetic``. One mixin per module keeps optional-dependency
isolation: this module imports only the DuckDB backend.
"""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.duckdb_numeric_source import (
    duckdb_non_numeric_descriptor,
)


class DuckdbArithmeticMixin:
    @classmethod
    def _non_numeric_descriptor(cls, data: DuckdbRelation, source_col: str) -> object | None:
        return duckdb_non_numeric_descriptor(data, source_col)
