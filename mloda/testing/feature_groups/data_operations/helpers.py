"""Shared helpers for data-operations tests.

Provides:
- ``extract_column``: Extract a column from any framework result as a Python list.
- ``make_feature_set``: Build a FeatureSet with optional partition_by/order_by.
- ``feature_set_for``: Build a FeatureSet around an Options that already exists.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.provider import FeatureSet
from mloda.user import Feature, Options


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
    **extra_context: Any,
) -> FeatureSet:
    """Build a FeatureSet with optional partition_by, order_by, mask, and extra context.

    Any additional keyword arguments are merged into the same Options context dict
    used by the explicit ``partition_by`` / ``order_by`` / ``mask`` arguments,
    enabling callers to pass operation-specific keys (e.g. ``constant=5``) without
    constructing ``Feature``/``Options`` manually. The explicit keyword arguments
    take precedence over ``extra_context`` on key collision.
    """
    context: dict[str, Any] = dict(extra_context)
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


def feature_set_for(feature_name: str, options: Options) -> FeatureSet:
    """Build a FeatureSet holding one feature that carries ``options`` verbatim.

    ``make_feature_set`` assembles the Options from keyword arguments; this one takes an
    Options that has already been assembled, which is the shape ``compute_values`` in the
    scalar-arity harness hands to a family.
    """
    fs = FeatureSet()
    fs.add(Feature(feature_name, options=options))
    return fs
