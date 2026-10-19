"""PyArrow implementation of key-lookup (production AND reference)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
)


def _source_family(arrow_type: pa.DataType) -> str:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "str"
    # uint64 cannot be compared at int64 width.
    if pa.types.is_integer(arrow_type) and not pa.types.is_uint64(arrow_type):
        return "int"
    return "other"


class PyArrowLookup(LookupFeatureGroup):
    """PyArrow backend; also the cross-framework reference implementation."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _compute_lookup(
        cls,
        data: pa.Table,
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> pa.Table:
        source = data.column(source_col)
        assert_key_family(feature_name, source_col, mapping, _source_family(source.type), str(source.type))

        values = pa.array(list(mapping.values()))
        # Compare ints at int64 width so keys outside the source width never match.
        probe = source.cast(pa.int64()) if pa.types.is_integer(source.type) else source
        keys = pa.array(list(mapping.keys()), type=probe.type)
        index = pc.index_in(probe, value_set=keys)
        mapped = pc.take(values, index)
        unmatched = pc.and_(pc.is_valid(source), pc.is_null(index))

        if policy == "raise":
            rows = pc.sum(unmatched).as_py() or 0
            if rows:
                distinct = pc.unique(pc.filter(source, unmatched)).to_pylist()
                raise build_unmatched_error(feature_name, rows, distinct, reasons)
        elif policy == "keep":
            mapped = pc.if_else(unmatched, source.cast(values.type), mapped)

        return data.append_column(feature_name, mapped)
