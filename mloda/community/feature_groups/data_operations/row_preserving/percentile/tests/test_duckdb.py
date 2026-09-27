"""Tests for DuckdbPercentile compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.duckdb_percentile import (
    DuckdbPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestDuckdbPercentile(DuckdbTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbPercentile


class TestDuckdbPercentileColumnCaseMismatch:
    """A source column whose case differs from the feature name must still resolve.

    DuckDB binds identifiers case-insensitively, so ``val`` in the feature name
    and ``Val`` in the table refer to the same column at the SQL level; the
    column-type lookup used to wrap NaN as null must resolve it too.
    """

    def test_case_mismatched_column_computes_without_error(self) -> None:
        import duckdb
        import pyarrow as pa
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core

        con = duckdb.connect(":memory:")
        pin_connection_utc_via_core(con)
        table = pa.table(
            {
                "grp": ["A", "A", "A"],
                "Val": pa.array([1.0, float("nan"), 3.0], type=pa.float64()),
            }
        )
        rel = DuckdbRelation.from_arrow(con, table)
        fs = make_feature_set("val__p50_percentile", partition_by=["grp"])

        result = DuckdbPercentile.calculate_feature(rel, fs)

        assert result.to_arrow_table().num_rows == 3
