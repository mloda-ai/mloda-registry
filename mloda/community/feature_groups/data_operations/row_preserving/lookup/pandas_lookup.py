"""Pandas implementation of key-lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
    value_family,
)

_VALUE_DTYPES: dict[str, str] = {"str": "object", "int": "Int64", "float": "float64", "bool": "boolean"}


def _source_family(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    # Nulls turn an int column into float64; integral values still count as int.
    if pd.api.types.is_float_dtype(dtype) and bool((series.dropna() % 1 == 0).all()):
        return "int"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "str"
    return "other"


class PandasLookup(LookupFeatureGroup):
    """Pandas backend: ``Series.map`` gathers values, ``isin`` flags unmatched keys."""

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
        assert_key_family(feature_name, source_col, mapping, family)
        if family == "int":
            source = source.astype("Int64")

        mapped = source.map(dict(mapping), na_action="ignore").astype(_VALUE_DTYPES[value_family(mapping)])
        unmatched = source.notna() & ~source.isin(list(mapping))

        if policy == "raise":
            rows = int(unmatched.sum())
            if rows:
                raise build_unmatched_error(feature_name, rows, source[unmatched].unique().tolist(), reasons)
        elif policy == "keep":
            mapped = mapped.where(~unmatched, source)

        return data.assign(**{feature_name: mapped})
