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
    """Case-mismatched source column names are rejected up front on every backend."""

    def test_case_mismatched_column_rejected(self) -> None:
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

        with pytest.raises(ValueError, match=r"Source column 'val' is not present"):
            DuckdbPercentile.calculate_feature(rel, fs)
