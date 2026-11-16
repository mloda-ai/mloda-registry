"""Tests for DuckdbPercentile compute implementation."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.core.abstract_plugins.components.data_types import DataType

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.duckdb_percentile import (
    DuckdbPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestDuckdbPercentile(DuckdbTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbPercentile


class TestDuckdbPercentileColumnCaseMismatch:
    """Case-mismatched source column names are rejected up front on every backend."""

    def test_case_mismatched_column_rejected(self) -> None:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        from mloda.testing.feature_groups.data_operations.helpers import (
            arrow_table_of,
            calculate_with,
            make_feature_set,
        )
        from mloda.testing.feature_groups.data_operations.mixins.frameworks import duckdb_test_connection

        con = duckdb_test_connection()
        try:
            table = TypedTestData(
                {"grp": (DataType.STRING, ["A", "A", "A"]), "Val": (DataType.DOUBLE, [1.0, float("nan"), 3.0])}
            )
            rel = DuckdbRelation.from_arrow(con, arrow_table_of(table))
            fs = make_feature_set("val__p50_percentile", partition_by=["grp"])

            with pytest.raises(ValueError, match=r"Source column 'val' is not present"):
                calculate_with(DuckdbPercentile, rel, fs)
        finally:
            con.close()
