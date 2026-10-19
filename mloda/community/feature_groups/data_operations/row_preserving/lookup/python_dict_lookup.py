"""PythonDict implementation of key-lookup (plain dict lookup per row)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
)


def _source_family(values: list[Any]) -> str | None:
    """``str``, ``int`` or ``other`` from the non-null values; None when the column has none."""
    families = {
        "str"
        if isinstance(v, str)
        else "int"
        if isinstance(v, int) and not isinstance(v, bool) and -(2**63) <= v < 2**63
        else "other"
        for v in values
        if v is not None
    }
    if not families:
        return None
    return families.pop() if len(families) == 1 else "other"


class PythonDictLookup(LookupFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_lookup(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        mapping: Mapping[Any, Any],
        policy: str,
        reasons: Mapping[Any, str],
    ) -> dict[str, list[Any]]:
        source = data[source_col]
        family = _source_family(source)
        if family is not None:
            types = ", ".join(sorted({type(v).__name__ for v in source if v is not None}))
            assert_key_family(feature_name, source_col, mapping, family, types)

        if policy == "raise":
            unmatched = [v for v in source if v is not None and v not in mapping]
            if unmatched:
                raise build_unmatched_error(feature_name, len(unmatched), set(unmatched), reasons)

        keep = policy == "keep"
        result = dict(data)
        result[feature_name] = [mapping.get(v, v if keep else None) if v is not None else None for v in source]
        return result
