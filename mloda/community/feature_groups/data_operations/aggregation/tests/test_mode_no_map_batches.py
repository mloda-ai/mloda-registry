"""Regression test: Polars Lazy aggregation must not use map_batches for mode.

Behavior-level guard: drives the mode path through ``calculate_feature`` and
inspects the resulting LazyFrame's query plan (``.explain()``). The plan must
not reference ``map_batches`` (Python per-group callback) nor the legacy
``_mode_with_insertion_order`` helper. This catches regressions even if the
callback is moved to a different module.
"""

from __future__ import annotations

import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
    PolarsLazyAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


def test_polars_lazy_aggregation_mode_plan_has_no_map_batches() -> None:
    """The mode aggregation plan must stay fully expression-based."""
    lf = pl.LazyFrame(
        {
            "region": ["A", "A", "A", "B", "B", "B"],
            "value_int": [10, 10, 20, 30, 30, 40],
        }
    )
    fs = make_feature_set("value_int__mode_agg", ["region"])

    result = PolarsLazyAggregation.calculate_feature(lf, fs)

    assert isinstance(result, pl.LazyFrame)

    plan = result.explain()
    assert "map_batches" not in plan, (
        f"Polars lazy mode aggregation plan must not contain map_batches (per-group Python callback); got plan:\n{plan}"
    )
    assert "_mode_with_insertion_order" not in plan, (
        "Polars lazy mode aggregation plan must not reference the legacy "
        f"_mode_with_insertion_order helper; got plan:\n{plan}"
    )

    # Sanity: the plan actually computed the mode and produces the expected values.
    collected = result.collect()
    result_map = dict(zip(collected["region"].to_list(), collected["value_int__mode_agg"].to_list()))
    assert result_map["A"] == 10
    assert result_map["B"] == 30
