"""Issue #717 cross-framework conditional-aggregation parity.

Dedicated fixture so the shared canonical data remains unchanged.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")
pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import DuckdbAggregation
from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import PolarsLazyAggregation
from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import SqliteAggregation
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


def _normalize(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _rows(result: Any, framework: str) -> list[tuple[Any, Any]]:
    if framework == "polars":
        table = result.collect().to_arrow()
    else:
        table = result.to_arrow_table()
    groups = table.column("group_key").to_pylist()
    values = table.column("masked_sum").to_pylist()
    return sorted((g, _normalize(v)) for g, v in zip(groups, values))


def test_issue_717_conditional_aggregation_nan_null_parity() -> None:
    """NaN/null must not satisfy greater_equal; all three frameworks agree."""
    arrow = pa.table(
        {
            "group_key": ["A", "A", "A", "B", "B"],
            "metric": pa.array([1.0, float("nan"), None, 3.0, 5.0], type=pa.float64()),
            "payload": [10, 20, 30, 40, 50],
        }
    )
    fs = make_feature_set(
        "masked_sum",
        ["group_key"],
        mask=("metric", "greater_equal", 2.0),
        in_features="payload",
        aggregation_type="sum",
    )

    dcon = duckdb.connect()
    scon = sqlite3.connect(":memory:")
    try:
        duck = DuckdbAggregation.calculate_feature(DuckdbRelation.from_arrow(dcon, arrow), fs)
        sqlite = SqliteAggregation.calculate_feature(SqliteRelation.from_arrow(scon, arrow), fs)
        polars = PolarsLazyAggregation.calculate_feature(pl.from_arrow(arrow).lazy(), fs)

        expected = [("A", None), ("B", 90)]
        assert _rows(duck, "duckdb") == expected
        assert _rows(sqlite, "sqlite") == expected
        assert _rows(polars, "polars") == expected
    finally:
        dcon.close()
        scon.close()
