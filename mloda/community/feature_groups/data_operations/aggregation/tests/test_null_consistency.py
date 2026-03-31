"""Cross-framework null consistency tests for column aggregation.

All test logic is in AggregationNullConsistencyBase. This file provides
the 5-framework adapter fixture and a thin subclass.

See: https://github.com/mloda-ai/mloda-registry/issues/74
"""

from __future__ import annotations

import sqlite3
from typing import Any, Generator

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
    DuckdbColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
    PandasColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
    PolarsLazyColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowColumnAggregation,
)
from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
    SqliteColumnAggregation,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.aggregation_null_consistency import (
    AggregationNullConsistencyBase,
)
from mloda.testing.feature_groups.data_operations.null_consistency import FrameworkAdapter
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


# ---------------------------------------------------------------------------
# Extract helpers
# ---------------------------------------------------------------------------


def _extract_pyarrow(result: Any, col: str) -> list[Any]:
    return list(result.column(col).to_pylist())


def _extract_pandas(result: Any, col: str) -> list[Any]:
    return [None if pd.isna(v) else v for v in result[col].tolist()]


def _extract_polars(result: Any, col: str) -> list[Any]:
    return list(result.collect()[col].to_list())


def _extract_arrow_backed(result: Any, col: str) -> list[Any]:
    return list(result.to_arrow_table().column(col).to_pylist())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_adapters() -> Generator[list[FrameworkAdapter], None, None]:
    """All 5 framework adapters for column aggregation."""
    arrow_table = PyArrowDataOpsTestDataCreator.create()
    conn_duckdb = duckdb.connect()
    conn_sqlite = sqlite3.connect(":memory:")

    adapters = [
        FrameworkAdapter("pyarrow", PyArrowColumnAggregation, arrow_table, _extract_pyarrow),
        FrameworkAdapter("pandas", PandasColumnAggregation, arrow_table.to_pandas(), _extract_pandas),
        FrameworkAdapter("polars", PolarsLazyColumnAggregation, pl.from_arrow(arrow_table).lazy(), _extract_polars),
        FrameworkAdapter(
            "duckdb",
            DuckdbColumnAggregation,
            DuckdbRelation.from_arrow(conn_duckdb, arrow_table),
            _extract_arrow_backed,
        ),
        FrameworkAdapter(
            "sqlite",
            SqliteColumnAggregation,
            SqliteRelation.from_arrow(conn_sqlite, arrow_table),
            _extract_arrow_backed,
        ),
    ]

    yield adapters

    conn_duckdb.close()
    conn_sqlite.close()


# ---------------------------------------------------------------------------
# Test class (all tests inherited from base)
# ---------------------------------------------------------------------------


class TestAggregationNullConsistency(AggregationNullConsistencyBase):
    """All tests inherited from AggregationNullConsistencyBase."""
