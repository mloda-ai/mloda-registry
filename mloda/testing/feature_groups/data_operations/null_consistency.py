"""Cross-framework null consistency testing infrastructure.

Provides ``FrameworkAdapter`` and comparison helpers for verifying
all framework implementations produce identical results on null
edge cases. PyArrow serves as the golden reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature


@dataclass
class FrameworkAdapter:
    """Wraps a single framework implementation for cross-framework comparison."""

    name: str
    impl_class: Any
    data: Any
    extract_fn: Callable[[Any, str], list[Any]]

    def calculate(self, feature_set: FeatureSet) -> Any:
        """Run calculate_feature on this framework's data."""
        return self.impl_class.calculate_feature(self.data, feature_set)

    def extract(self, result: Any, column_name: str) -> list[Any]:
        """Extract a named column from a calculate result."""
        return self.extract_fn(result, column_name)


def make_feature_set(feature_name: str, context: dict[str, Any] | None = None) -> FeatureSet:
    """Build a FeatureSet with optional context options."""
    options = Options(context=context) if context else Options()
    feature = Feature(feature_name, options=options)
    fs = FeatureSet()
    fs.add(feature)
    return fs


def _assert_values_match(
    ref_val: Any,
    cand_val: Any,
    label: str,
    ref_name: str,
    cand_name: str,
    use_approx: bool,
    rel: float,
) -> None:
    """Assert a single reference value matches a candidate value."""
    if ref_val is None:
        assert cand_val is None, f"{label}: {cand_name}={cand_val}, {ref_name}=None"
    elif use_approx:
        assert cand_val is not None, f"{label}: {cand_name}=None, {ref_name}={ref_val}"
        assert cand_val == pytest.approx(ref_val, rel=rel), f"{label}: {cand_name}={cand_val} != {ref_name}={ref_val}"
    else:
        assert cand_val == ref_val, f"{label}: {cand_name}={cand_val} != {ref_name}={ref_val}"


def assert_all_frameworks_agree(
    adapters: list[FrameworkAdapter],
    feature_set: FeatureSet,
    feature_name: str,
    use_approx: bool = False,
    rel: float = 1e-6,
) -> dict[str, list[Any]]:
    """Run a broadcast feature across all adapters, assert row-level equality.

    Uses the first adapter (PyArrow) as the golden reference.
    Returns {name: column_values} for further inspection.
    """
    results: dict[str, list[Any]] = {}
    for adapter in adapters:
        raw = adapter.calculate(feature_set)
        results[adapter.name] = adapter.extract(raw, feature_name)

    ref_name = adapters[0].name
    ref_col = results[ref_name]

    for adapter in adapters[1:]:
        cand_col = results[adapter.name]
        assert len(cand_col) == len(ref_col), f"{adapter.name}: {len(cand_col)} rows vs {ref_name}: {len(ref_col)}"
        for i, (rv, cv) in enumerate(zip(ref_col, cand_col)):
            _assert_values_match(rv, cv, f"row {i}", ref_name, adapter.name, use_approx, rel)

    return results


def assert_all_frameworks_agree_grouped(
    adapters: list[FrameworkAdapter],
    feature_set: FeatureSet,
    feature_name: str,
    group_key: str,
    use_approx: bool = False,
    rel: float = 1e-6,
) -> dict[str, dict[Any, Any]]:
    """Run a grouped feature across all adapters, assert equality by group key.

    Handles different row orderings by comparing {group_key: value} maps.
    Returns {adapter_name: {key: value}} for further inspection.
    """
    results: dict[str, dict[Any, Any]] = {}
    for adapter in adapters:
        raw = adapter.calculate(feature_set)
        keys = adapter.extract(raw, group_key)
        vals = adapter.extract(raw, feature_name)
        results[adapter.name] = dict(zip(keys, vals))

    ref_name = adapters[0].name
    ref_map = results[ref_name]

    for adapter in adapters[1:]:
        cand_map = results[adapter.name]
        assert set(cand_map.keys()) == set(ref_map.keys()), (
            f"{adapter.name} groups {set(cand_map.keys())} != {ref_name} groups {set(ref_map.keys())}"
        )
        for key in ref_map:
            _assert_values_match(
                ref_map[key],
                cand_map[key],
                f"group {key!r}",
                ref_name,
                adapter.name,
                use_approx,
                rel,
            )

    return results
