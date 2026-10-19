"""Pandas implementation of key-lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
    key_family,
    value_family,
)

_VALUE_DTYPES: dict[str, str] = {"str": "object", "int": "Int64", "float": "float64", "bool": "boolean"}


def _source_family(series: pd.Series) -> str | None:
    """``str``, ``int`` or ``other``; None when an object column holds no non-null values."""
    dtype = series.dtype
    if str(dtype).lower() == "uint64":
        return "other"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    # Nulls turn an int column into float64; integral finite values within int64 still count as int.
    if pd.api.types.is_float_dtype(dtype):
        values = series.dropna()
        integral = bool(np.isfinite(values).all()) and bool((values % 1 == 0).all())
        return "int" if integral and bool((values.abs() < 2**63).all()) else "other"
    if pd.api.types.is_object_dtype(dtype):
        inferred = pd.api.types.infer_dtype(series, skipna=True)
        if inferred == "empty":
            return None
        if inferred == "integer":
            return "int" if bool(series.dropna().between(-(2**63), 2**63 - 1).all()) else "other"
        return "str" if inferred == "string" else "other"
    if pd.api.types.is_string_dtype(dtype):
        return "str"
    return "other"


class PandasLookup(LookupFeatureGroup):
    """Pandas backend: an exact index lookup gathers values and flags unmatched keys."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_lookup(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> pd.DataFrame:
        source = data[source_col]
        family = _source_family(source)
        if family is None:
            empty = pd.Series(None, index=data.index, dtype=_VALUE_DTYPES[value_family(mapping)])
            return data.assign(**{feature_name: empty})
        if not (source.isna().all() and pd.api.types.is_float_dtype(source.dtype)):
            assert_key_family(feature_name, source_col, mapping, family, str(source.dtype))
        if family == "int":
            source = source.astype("Int64")

        # Index gather keeps ints exact (Series.map round-trips nullable ints through float64).
        key_index = pd.Index(list(mapping), dtype="Int64" if key_family(mapping) == "int" else None)
        indexer = key_index.get_indexer(source.array)
        values = pd.array(list(mapping.values()), dtype=_VALUE_DTYPES[value_family(mapping)])
        mapped = pd.Series(values.take(indexer, allow_fill=True), index=data.index).astype(values.dtype)
        unmatched = source.notna() & pd.Series(indexer == -1, index=data.index)

        if policy == "raise":
            rows = int(unmatched.sum())
            if rows:
                raise build_unmatched_error(feature_name, rows, source[unmatched].unique().tolist(), reasons)
        elif policy == "keep":
            mapped = mapped.where(~unmatched, source)

        return data.assign(**{feature_name: mapped})
