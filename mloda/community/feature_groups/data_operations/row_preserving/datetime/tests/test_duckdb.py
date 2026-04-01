"""Tests for DuckdbDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.datetime.duckdb_datetime import (
    DuckdbDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.helpers import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime import (
    DateTimeTestBase,
)


class TestDuckdbDateTimeExtraction(DuckdbTestMixin, DateTimeTestBase):
    """Standard tests inherited from the base class."""

    def setup_method(self) -> None:
        self.conn = duckdb.connect()
        self.conn.execute("SET timezone = 'UTC'")
        super(DuckdbTestMixin, self).setup_method()

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbDateTimeExtraction


class TestDuckdbTimezoneBoundary:
    """Tests for DuckDB timezone-sensitive datetime extraction.

    DuckDB evaluates date-part extraction for TIMESTAMPTZ in the session
    timezone. These tests use non-midnight UTC timestamps near day boundaries
    to verify that the DuckDB implementation produces correct UTC results
    regardless of session timezone settings.
    """

    def setup_method(self) -> None:
        self.conn = duckdb.connect()
        self.conn.execute("SET timezone = 'UTC'")

    def teardown_method(self) -> None:
        self.conn.close()

    def _make_boundary_data(self) -> DuckdbRelation:
        """Create a table with timestamps near midnight boundaries (UTC)."""
        table = pa.table(
            {
                "timestamp": pa.array(
                    [
                        datetime(2023, 6, 15, 23, 59, 59, tzinfo=timezone.utc),
                        datetime(2023, 6, 16, 0, 0, 1, tzinfo=timezone.utc),
                        datetime(2023, 12, 31, 23, 30, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        return DuckdbRelation.from_arrow(self.conn, table)

    def _make_feature_set(self, feature_name: str) -> FeatureSet:
        feature = Feature(feature_name, options=Options())
        fs = FeatureSet()
        fs.add(feature)
        return fs

    def _extract_column(self, result: DuckdbRelation, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def test_day_near_midnight(self) -> None:
        """Day extraction near midnight should reflect UTC day, not local day."""
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__day")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        days = self._extract_column(result, "timestamp__day")
        assert days == [15, 16, 31, 1]

    def test_hour_near_midnight(self) -> None:
        """Hour extraction near midnight should reflect UTC hour."""
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__hour")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        hours = self._extract_column(result, "timestamp__hour")
        assert hours == [23, 0, 23, 0]

    def test_year_at_year_boundary(self) -> None:
        """Year extraction at year boundary should reflect UTC year."""
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__year")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        years = self._extract_column(result, "timestamp__year")
        assert years == [2023, 2023, 2023, 2024]

    def test_month_at_year_boundary(self) -> None:
        """Month extraction at year boundary should reflect UTC month."""
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__month")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        months = self._extract_column(result, "timestamp__month")
        assert months == [6, 6, 12, 1]

    def test_dayofweek_near_midnight(self) -> None:
        """Day-of-week near midnight should reflect UTC day, not local day.

        2023-06-15 (Thu=3), 2023-06-16 (Fri=4), 2023-12-31 (Sun=6), 2024-01-01 (Mon=0)
        """
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__dayofweek")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        dows = self._extract_column(result, "timestamp__dayofweek")
        assert dows == [3, 4, 6, 0]

    def test_quarter_at_year_boundary(self) -> None:
        """Quarter extraction at year boundary should reflect UTC quarter."""
        data = self._make_boundary_data()
        fs = self._make_feature_set("timestamp__quarter")
        result = DuckdbDateTimeExtraction.calculate_feature(data, fs)
        quarters = self._extract_column(result, "timestamp__quarter")
        assert quarters == [2, 2, 4, 1]
