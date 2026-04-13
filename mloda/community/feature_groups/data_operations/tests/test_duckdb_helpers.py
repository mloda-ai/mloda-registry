"""Tests for the DuckDB relation helpers."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.duckdb_helpers import (
    group_aggregate,
    query_with_alias,
)


@pytest.fixture()
def sample_relation() -> DuckdbRelation:
    """Small relation with a repeating key column for group-by tests."""
    conn = duckdb.connect(":memory:")
    rel = DuckdbRelation.from_dict(
        conn,
        {
            "key": ["a", "a", "b", "b", "b"],
            "value": [1, 2, 3, 4, 5],
        },
    )
    return rel


class TestGroupAggregate:
    def test_returns_duckdb_relation(self, sample_relation: DuckdbRelation) -> None:
        """group_aggregate must return a DuckdbRelation, not a raw DuckDBPyRelation."""
        result = group_aggregate(sample_relation, '"key", SUM("value") AS total', '"key"')
        assert isinstance(result, DuckdbRelation)

    def test_reuses_same_connection(self, sample_relation: DuckdbRelation) -> None:
        """The helper must not open a new connection."""
        result = group_aggregate(sample_relation, '"key", SUM("value") AS total', '"key"')
        assert result.connection is sample_relation.connection

    def test_group_by_sums_values(self, sample_relation: DuckdbRelation) -> None:
        """A SUM group-by returns one row per key with the correct total."""
        result = group_aggregate(sample_relation, '"key", SUM("value") AS total', '"key"')
        arrow = result.order('"key"').to_arrow_table()
        data = arrow.to_pydict()
        assert data["key"] == ["a", "b"]
        assert data["total"] == [3, 12]

    def test_does_not_mutate_source(self, sample_relation: DuckdbRelation) -> None:
        """Aggregating must not change the columns of the source relation."""
        group_aggregate(sample_relation, '"key", COUNT(*) AS n', '"key"')
        assert sample_relation.columns == ["key", "value"]


class TestQueryWithAlias:
    def test_returns_duckdb_relation(self, sample_relation: DuckdbRelation) -> None:
        """query_with_alias must return a DuckdbRelation."""
        result = query_with_alias(sample_relation, "__t", 'SELECT * FROM __t WHERE "value" > 2')
        assert isinstance(result, DuckdbRelation)

    def test_reuses_same_connection(self, sample_relation: DuckdbRelation) -> None:
        result = query_with_alias(sample_relation, "__t", "SELECT * FROM __t")
        assert result.connection is sample_relation.connection

    def test_select_filters_rows(self, sample_relation: DuckdbRelation) -> None:
        """Arbitrary SELECT against the alias works end to end."""
        result = query_with_alias(sample_relation, "__t", 'SELECT * FROM __t WHERE "value" > 2')
        arrow = result.order('"value"').to_arrow_table()
        data = arrow.to_pydict()
        assert data["value"] == [3, 4, 5]

    def test_window_function(self, sample_relation: DuckdbRelation) -> None:
        """The helper is specifically useful for window queries that project() cannot express directly."""
        sql = 'SELECT *, ROW_NUMBER() OVER (PARTITION BY "key" ORDER BY "value") AS rn FROM __t'
        result = query_with_alias(sample_relation, "__t", sql)
        arrow = result.order('"key"', '"value"').to_arrow_table()
        data = arrow.to_pydict()
        assert data["rn"] == [1, 2, 1, 2, 3]
