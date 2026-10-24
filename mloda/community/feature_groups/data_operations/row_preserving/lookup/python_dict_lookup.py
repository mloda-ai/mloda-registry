"""PythonDict implementation of key-lookup (plain dict lookup per row)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    LookupFeatureGroup,
    assert_key_family,
    build_unmatched_error,
    key_family,
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


def _type_label(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and not -(2**63) <= value < 2**63:
        return "int (outside int64)"
    return type(value).__name__


class PythonDictLookup(Implementation[LookupFeatureGroup]):
    feature_group = LookupFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[LookupFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.lookup(data, features, compute_lookup=cls._compute_lookup)

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
        if family is not None and family != key_family(mapping):
            types = ", ".join(sorted({_type_label(v) for v in source if v is not None}))
            assert_key_family(feature_name, source_col, mapping, family, types)

        if policy == "raise":
            unmatched = [v for v in source if v is not None and v not in mapping]
            if unmatched:
                raise build_unmatched_error(feature_name, len(unmatched), set(unmatched), reasons)

        keep = policy == "keep"
        result = dict(data)
        result[feature_name] = [mapping.get(v, v if keep else None) if v is not None else None for v in source]
        return result
