"""PyArrow test data creator for data operations plugins."""

from __future__ import annotations

import functools
from typing import Any

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.data_creator.base import DataOperationsTestDataCreator


class ColumnSelectiveMixin:
    """Mixin for a data creator: serve only the requested columns, like a column-pruning reader."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        full = super().calculate_feature(data, features)  # type: ignore[misc]
        names = list(features.get_all_names())
        if isinstance(full, dict):
            return {name: full[name] for name in names}
        if hasattr(full, "select"):
            return full.select(names)
        return full[names]


@functools.cache
def column_selective(creator: type) -> type:
    """Return ``creator`` narrowed to the requested columns (any dict, pyarrow or pandas creator)."""
    return type(f"ColumnSelective{creator.__name__}", (ColumnSelectiveMixin, creator), {})


class PyArrowDataOpsTestDataCreator(DataOperationsTestDataCreator):
    compute_framework = PyArrowTable

    # Columns that are all-null need an explicit type so PyArrow produces a
    # typed array (e.g. float64) instead of pa.null().  Operations like
    # hash_stddev / hash_variance / hash_first_last raise
    # ArrowNotImplementedError on pa.null()-typed columns.
    EXPLICIT_TYPES: dict[str, Any] = {"score": "float64"}

    @classmethod
    def create(cls) -> Any:
        import pyarrow as pa

        raw = cls.get_raw_data()
        arrays: list[pa.Array] = []
        names: list[str] = []
        for col_name, values in raw.items():
            names.append(col_name)
            explicit = cls.EXPLICIT_TYPES.get(col_name)
            if explicit is not None:
                arrays.append(pa.array(values, type=getattr(pa, explicit)()))
            else:
                arrays.append(pa.array(values))
        return pa.table(dict(zip(names, arrays)))
