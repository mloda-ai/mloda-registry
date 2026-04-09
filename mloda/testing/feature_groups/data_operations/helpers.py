"""Shared helpers for data-operations tests.

Provides:
- ``extract_column``: Extract a column from any framework result as a Python list.
- ``make_feature_set``: Build a FeatureSet with optional partition_by/order_by.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list.

    Handles pa.Table (direct .column() access), relation types
    (DuckdbRelation, SqliteRelation) that expose .to_arrow_table(),
    Polars LazyFrames that expose .collect(), and pandas DataFrames.
    """
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    if hasattr(result, "to_arrow_table"):
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())
    if hasattr(result, "collect"):
        df = result.collect()
        return list(df[column_name].to_list())
    return list(result[column_name])


def make_feature_set(
    feature_name: str,
    partition_by: list[str] | None = None,
    order_by: str | None = None,
    mask: tuple[Any, ...] | list[tuple[Any, ...]] | None = None,
) -> FeatureSet:
    """Build a FeatureSet with optional partition_by, order_by, and mask options."""
    context: dict[str, Any] = {}
    if partition_by is not None:
        context["partition_by"] = partition_by
    if order_by is not None:
        context["order_by"] = order_by
    if mask is not None:
        context["mask"] = mask
    feature = Feature(feature_name, options=Options(context=context))
    fs = FeatureSet()
    fs.add(feature)
    return fs
