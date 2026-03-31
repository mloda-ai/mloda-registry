"""Shared test base class for cross-framework null consistency in group aggregation.

All test logic lives here. Framework-specific test files subclass and
provide an ``all_adapters`` fixture that yields a list of FrameworkAdapter
objects (one per framework under test).

Expected values are from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11).

See: https://github.com/mloda-ai/mloda-registry/issues/74
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.testing.feature_groups.data_operations.null_consistency import (
    FrameworkAdapter,
    assert_all_frameworks_agree_grouped,
    make_feature_set,
)


class GroupAggregationNullConsistencyBase:
    """Abstract base for group aggregation null consistency tests.

    Subclasses must ensure an ``all_adapters`` pytest fixture is available
    that yields ``list[FrameworkAdapter]`` with PyArrow as the first entry.
    """

    # -- Null group keys --------------------------------------------------------

    def test_null_key_group_exists_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Row 11 has region=None, value_int=-10. The None group sum should be -10."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__sum_grouped", "region")
        assert results[all_adapters[0].name][None] == -10

    def test_null_key_group_exists_count(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has 1 non-null value_int. Count should be 1."""
        fs = make_feature_set("value_int__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__count_grouped", "region")
        assert results[all_adapters[0].name][None] == 1

    def test_null_key_group_exists_min(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has value_int=-10. Min should be -10."""
        fs = make_feature_set("value_int__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__min_grouped", "region")
        assert results[all_adapters[0].name][None] == -10

    def test_null_key_group_exists_max(self, all_adapters: list[FrameworkAdapter]) -> None:
        """None group has value_int=-10. Max should be -10."""
        fs = make_feature_set("value_int__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__max_grouped", "region")
        assert results[all_adapters[0].name][None] == -10

    def test_group_count_includes_null_key(self, all_adapters: list[FrameworkAdapter]) -> None:
        """There should be exactly 4 groups: A, B, C, and None."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            keys = adapter.extract(raw, "region")
            assert len(keys) == 4, f"{adapter.name}: expected 4 groups, got {len(keys)}"

    # -- Partial-null values within groups -------------------------------------

    def test_group_b_sum_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B sum: 50+30+60=140 (null at row 4 skipped)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__sum_grouped", "region")
        assert results[all_adapters[0].name]["B"] == 140

    def test_group_b_count_excludes_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B count: 3 non-null values (null at row 4 excluded)."""
        fs = make_feature_set("value_int__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__count_grouped", "region")
        assert results[all_adapters[0].name]["B"] == 3

    def test_group_b_avg_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B avg: 140/3 (null at row 4 skipped)."""
        fs = make_feature_set("value_int__avg_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(
            all_adapters,
            fs,
            "value_int__avg_grouped",
            "region",
            use_approx=True,
        )
        assert results[all_adapters[0].name]["B"] == pytest.approx(140.0 / 3.0, rel=1e-6)

    def test_group_b_min_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B min: 30 (null excluded)."""
        fs = make_feature_set("value_int__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__min_grouped", "region")
        assert results[all_adapters[0].name]["B"] == 30

    def test_group_b_max_skips_null(self, all_adapters: list[FrameworkAdapter]) -> None:
        """B max: 60 (null excluded)."""
        fs = make_feature_set("value_int__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "value_int__max_grouped", "region")
        assert results[all_adapters[0].name]["B"] == 60

    # -- All-null column across groups -----------------------------------------

    def test_all_null_sum_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Sum of all-null column per group should be None for every group."""
        fs = make_feature_set("score__sum_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__sum_grouped", "region")
        for key, val in results[all_adapters[0].name].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_count_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Count of all-null column per group should be 0 for every group."""
        fs = make_feature_set("score__count_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__count_grouped", "region")
        for key, val in results[all_adapters[0].name].items():
            assert val == 0, f"group {key!r}: expected 0, got {val}"

    def test_all_null_min_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Min of all-null column per group should be None."""
        fs = make_feature_set("score__min_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__min_grouped", "region")
        for key, val in results[all_adapters[0].name].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_max_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Max of all-null column per group should be None."""
        fs = make_feature_set("score__max_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__max_grouped", "region")
        for key, val in results[all_adapters[0].name].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    def test_all_null_avg_per_group(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Avg of all-null column per group should be None."""
        fs = make_feature_set("score__avg_grouped", context={"partition_by": ["region"]})
        results = assert_all_frameworks_agree_grouped(all_adapters, fs, "score__avg_grouped", "region")
        for key, val in results[all_adapters[0].name].items():
            assert val is None, f"group {key!r}: expected None, got {val}"

    # -- Multi-key partition with null keys ------------------------------------

    def test_multi_key_null_category_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (B, None) should have sum=30 (row 6: value_int=30)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[("B", None)] == 30, (
                f"{adapter.name}: group (B, None) sum={result_map.get(('B', None))}, expected 30"
            )

    def test_multi_key_null_region_sum(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (None, X) should have sum=-10 (row 11: value_int=-10)."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[(None, "X")] == -10, (
                f"{adapter.name}: group (None, X) sum={result_map.get((None, 'X'))}, expected -10"
            )

    def test_multi_key_group_with_null_value(self, all_adapters: list[FrameworkAdapter]) -> None:
        """Group (B, X) has rows 4 (null) and 7 (60). Sum should be 60."""
        fs = make_feature_set("value_int__sum_grouped", context={"partition_by": ["region", "category"]})
        for adapter in all_adapters:
            raw = adapter.calculate(fs)
            region_col = adapter.extract(raw, "region")
            category_col = adapter.extract(raw, "category")
            value_col = adapter.extract(raw, "value_int__sum_grouped")
            result_map = {(region_col[i], category_col[i]): value_col[i] for i in range(len(region_col))}
            assert result_map[("B", "X")] == 60, (
                f"{adapter.name}: group (B, X) sum={result_map.get(('B', 'X'))}, expected 60"
            )
