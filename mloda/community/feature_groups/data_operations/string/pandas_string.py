"""Pandas implementation for string operation feature groups."""

from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


def _pyarrow_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _is_pyarrow_backed(col: Any) -> bool:
    dtype = col.dtype
    if isinstance(dtype, pd.ArrowDtype):
        return True
    return isinstance(dtype, pd.StringDtype) and dtype.storage in ("pyarrow", "pyarrow_numpy")


def _pyarrow_backed(col: Any, op: str) -> Any:
    """Return string-valued columns as pyarrow-backed so case mapping matches the reference."""
    if _is_pyarrow_backed(col):
        return col
    if pd.api.types.infer_dtype(col, skipna=True) not in ("string", "empty"):
        return col
    if not _pyarrow_available():
        raise ValueError(
            f"Pandas {op} on non-pyarrow string storage ({col.dtype}) requires pyarrow for simple case mapping. "
            "Install mloda-community-string[pandas] or use the PyArrow backend."
        )
    return col.astype("string[pyarrow]")


class PandasStringOps(Implementation[StringFeatureGroup]):
    feature_group = StringFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[StringFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.transform_string(data, features, compute_string=cls._compute_string)

    @classmethod
    def _compute_string(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]

        if op == "upper":
            data[feature_name] = _pyarrow_backed(col, op).str.upper()
        elif op == "lower":
            data[feature_name] = _pyarrow_backed(col, op).str.lower()
        elif op == "trim":
            data[feature_name] = col.str.strip()
        elif op == "length":
            data[feature_name] = col.str.len().astype("Int64")
        elif op == "reverse":
            data[feature_name] = col.str[::-1]
        else:
            raise ValueError(f"Unsupported string operation: {op}")

        return data
