"""Tests for DuckdbStringOps compute implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.string.duckdb_string import (
    DuckdbStringOps,
)


@pytest.fixture
def sample_relation() -> Any:
    conn = duckdb.connect()
    arrow_table = pa.table(
        {
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    relation = DuckdbRelation.from_arrow(conn, arrow_table)
    yield relation
    conn.close()


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestUpper:
    def test_upper_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("name__upper")
        result = DuckdbStringOps.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("name__upper").to_pylist()
        assert result_col == ["ALICE", "BOB", "CHARLIE", None, "EVE"]


class TestLength:
    def test_length_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("name__length")
        result = DuckdbStringOps.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("name__length").to_pylist()
        assert result_col == [5, 3, 7, None, 3]


class TestNullPropagation:
    def test_null_produces_null_for_upper(self, sample_relation: Any) -> None:
        fs = _make_feature_set("name__upper")
        result = DuckdbStringOps.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("name__upper").to_pylist()
        assert result_col[3] is None


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_relation: Any) -> None:
        fs = _make_feature_set("name__upper")
        result = DuckdbStringOps.calculate_feature(sample_relation, fs)

        assert result.to_arrow_table().num_rows == 5

    def test_result_type(self, sample_relation: Any) -> None:
        fs = _make_feature_set("name__upper")
        result = DuckdbStringOps.calculate_feature(sample_relation, fs)

        assert isinstance(result, DuckdbRelation)
