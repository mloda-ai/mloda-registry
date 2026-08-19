"""Unit tests for shared python_dict helper utilities (group keys, sorting, reductions)."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    group_key_value,
    is_nan,
    mode,
    nulls_last_sort_key,
    reduce_agg,
    values_equal,
    variance,
)


class TestIsNan:
    def test_true_for_float_nan(self) -> None:
        assert is_nan(float("nan")) is True

    def test_false_for_none(self) -> None:
        assert is_nan(None) is False

    def test_false_for_ordinary_float(self) -> None:
        assert is_nan(1.5) is False

    def test_false_for_infinity(self) -> None:
        """Infinity is a well-ordered float, not NaN."""
        assert is_nan(float("inf")) is False
        assert is_nan(float("-inf")) is False

    def test_true_for_decimal_nan(self) -> None:
        """is_nan uses the type-agnostic ``value != value`` test, so Decimal('NaN') counts too."""
        assert is_nan(Decimal("NaN")) is True

    def test_false_for_decimal_real_value(self) -> None:
        assert is_nan(Decimal("1")) is False

    def test_true_for_numpy_float32_nan(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_nan(np.float32("nan")) is True

    def test_true_for_numpy_float64_nan(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_nan(np.float64("nan")) is True

    def test_false_for_numpy_float32_real_value(self) -> None:
        np = pytest.importorskip("numpy")
        assert is_nan(np.float32(1.0)) is False

    def test_false_for_int(self) -> None:
        assert is_nan(5) is False

    def test_false_for_string(self) -> None:
        assert is_nan("nan") is False


class TestGroupKeyValue:
    def test_passes_through_non_nan_values(self) -> None:
        assert group_key_value(5) == 5
        assert group_key_value("a") == "a"
        assert group_key_value(None) is None

    def test_maps_distinct_nan_objects_to_the_same_sentinel(self) -> None:
        nan_a = float("nan")
        nan_b = float("nan")
        assert nan_a is not nan_b, "sanity check: these must be distinct NaN objects"
        assert group_key_value(nan_a) is group_key_value(nan_b)

    def test_sentinel_differs_from_any_real_value_and_from_none(self) -> None:
        sentinel = group_key_value(float("nan"))
        assert sentinel != 0
        assert sentinel is not None


class TestNullsLastSortKey:
    def test_none_and_nan_share_the_same_key(self) -> None:
        assert nulls_last_sort_key(None) == nulls_last_sort_key(float("nan"))

    def test_real_value_sorts_before_the_null_tier(self) -> None:
        assert nulls_last_sort_key(1) < nulls_last_sort_key(None)
        assert nulls_last_sort_key(1) < nulls_last_sort_key(float("nan"))

    def test_sorted_places_none_and_nan_last(self) -> None:
        values: list[Any] = [3, None, 1, float("nan"), 2]
        ordered = sorted(values, key=nulls_last_sort_key)
        assert ordered[:3] == [1, 2, 3]
        assert ordered[3] is None or is_nan(ordered[3])
        assert ordered[4] is None or is_nan(ordered[4])

    def test_sorted_never_raises_on_a_mixed_null_and_datetime_column(self) -> None:
        u = timezone.utc
        values: list[Any] = [
            datetime(2023, 1, 2, tzinfo=u),
            None,
            float("nan"),
            datetime(2023, 1, 1, tzinfo=u),
        ]
        ordered = sorted(values, key=nulls_last_sort_key)
        assert ordered[0] == datetime(2023, 1, 1, tzinfo=u)
        assert ordered[1] == datetime(2023, 1, 2, tzinfo=u)

    def test_sorted_never_raises_on_a_mixed_null_and_string_column(self) -> None:
        values: list[Any] = ["b", None, float("nan"), "a"]
        ordered = sorted(values, key=nulls_last_sort_key)
        assert ordered[0] == "a"
        assert ordered[1] == "b"

    def test_infinity_sorts_as_a_real_value_not_as_null(self) -> None:
        values: list[Any] = [float("inf"), None, 1.0, float("-inf")]
        ordered = sorted(values, key=nulls_last_sort_key)
        assert ordered[:3] == [float("-inf"), 1.0, float("inf")]

    def test_negative_zero_and_zero_sort_together(self) -> None:
        ordered = sorted([0.0, -0.0], key=nulls_last_sort_key)
        assert ordered[0] == ordered[1] == 0.0


class TestValuesEqual:
    def test_equal_for_identical_scalars(self) -> None:
        assert values_equal(1, 1) is True
        assert values_equal("a", "a") is True

    def test_not_equal_for_different_scalars(self) -> None:
        assert values_equal(1, 2) is False

    def test_nan_equals_nan(self) -> None:
        assert values_equal(float("nan"), float("nan")) is True

    def test_nan_does_not_equal_none(self) -> None:
        """None and NaN share nulls_last_sort_key's tier but are NOT equal under values_equal."""
        assert values_equal(float("nan"), None) is False

    def test_none_equals_none(self) -> None:
        assert values_equal(None, None) is True

    def test_infinity_equals_itself(self) -> None:
        assert values_equal(float("inf"), float("inf")) is True

    def test_negative_zero_equals_zero(self) -> None:
        assert values_equal(-0.0, 0.0) is True

    def test_tuple_recursion_equal_tuples(self) -> None:
        assert values_equal((1, "a"), (1, "a")) is True

    def test_tuple_recursion_different_length_tuples_are_not_equal(self) -> None:
        assert values_equal((1, 2), (1,)) is False

    def test_tuple_recursion_element_mismatch(self) -> None:
        assert values_equal((1, 2), (1, 3)) is False

    def test_tuple_recursion_nan_element_matches(self) -> None:
        """A NaN element inside a group-key tuple compares equal via recursion."""
        assert values_equal((1, float("nan")), (1, float("nan"))) is True

    def test_tuple_recursion_none_and_nan_elements_do_not_match(self) -> None:
        assert values_equal((None, 1), (float("nan"), 1)) is False


class TestMode:
    def test_clear_winner(self) -> None:
        assert mode([1, 1, 2]) == 1

    def test_tie_breaks_by_first_occurrence(self) -> None:
        assert mode([2, 5, 2, 5]) == 2

    def test_skips_none(self) -> None:
        assert mode([None, 7, 7, None, 9]) == 7

    def test_all_none_returns_none(self) -> None:
        assert mode([None, None]) is None

    def test_empty_returns_none(self) -> None:
        assert mode([]) is None


class TestVariance:
    def test_population_variance_matches_known_value(self) -> None:
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert variance(values, ddof=0, as_std=False) == pytest.approx(4.0)

    def test_population_std_is_sqrt_of_variance(self) -> None:
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert variance(values, ddof=0, as_std=True) == pytest.approx(2.0)

    def test_sample_variance_uses_ddof_1_matches_stdlib_statistics(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert variance(values, ddof=1, as_std=False) == pytest.approx(statistics.variance(values))

    def test_returns_none_when_too_few_values_for_ddof(self) -> None:
        assert variance([5.0], ddof=1, as_std=False) is None

    def test_returns_none_for_empty_population_variance(self) -> None:
        assert variance([], ddof=0, as_std=False) is None

    def test_single_value_population_variance_is_zero(self) -> None:
        assert variance([5.0], ddof=0, as_std=False) == pytest.approx(0.0)


class TestReduceAgg:
    def test_sum_skips_none(self) -> None:
        assert reduce_agg("sum", [1.0, None, 2.0]) == pytest.approx(3.0)

    def test_sum_all_none_returns_none(self) -> None:
        assert reduce_agg("sum", [None, None]) is None

    def test_avg_and_mean_are_equivalent(self) -> None:
        values = [1.0, 2.0, 3.0]
        assert reduce_agg("avg", values) == pytest.approx(2.0)
        assert reduce_agg("mean", values) == pytest.approx(2.0)

    def test_count_counts_non_null_only(self) -> None:
        assert reduce_agg("count", [1, None, 2, None]) == 2

    def test_min_skips_none_and_nan(self) -> None:
        assert reduce_agg("min", [float("nan"), None, 3.0, 1.0]) == pytest.approx(1.0)

    def test_max_skips_none_and_nan(self) -> None:
        assert reduce_agg("max", [float("nan"), None, 3.0, 1.0]) == pytest.approx(3.0)

    def test_min_all_nan_returns_none(self) -> None:
        assert reduce_agg("min", [float("nan"), float("nan")]) is None

    def test_first_and_last_skip_none(self) -> None:
        assert reduce_agg("first", [None, 5, 6]) == 5
        assert reduce_agg("last", [5, 6, None]) == 6

    def test_median_skips_none(self) -> None:
        assert reduce_agg("median", [None, 1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_unsupported_agg_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            reduce_agg("not_a_real_agg_type", [1, 2, 3])

    def test_nunique_distinct_nan_objects_matches_pyarrow_count_distinct(self) -> None:
        """DEFECT G: nunique must not depend on NaN object identity.

        ``Table.to_pylist()`` (the realistic path feeding this backend) yields distinct NaN
        objects, not one shared literal; ``reduce_agg`` builds a plain ``set()`` over raw
        values, so those distinct NaN objects each hash/compare as a separate element and
        nunique comes out one too high.
        """
        pa = pytest.importorskip("pyarrow")
        pc = pytest.importorskip("pyarrow.compute")

        values = [1.0, float("nan"), 3.0, float("nan")]
        assert values[1] is not values[3], "sanity check: these must be distinct NaN objects"

        oracle = pc.count_distinct(pa.array(values, type=pa.float64())).as_py()
        assert oracle == 3, f"sanity check on the oracle itself failed: {oracle!r}"

        result = reduce_agg("nunique", values)
        assert result == oracle, f"nunique={result!r} != PyArrow oracle count_distinct={oracle!r}"

    def test_mode_distinct_nan_objects_matches_pyarrow_mode(self) -> None:
        """DEFECT G: mode must not depend on NaN object identity either."""
        pa = pytest.importorskip("pyarrow")
        pc = pytest.importorskip("pyarrow.compute")

        values = [1.0, float("nan"), 3.0, float("nan")]
        assert values[1] is not values[3], "sanity check: these must be distinct NaN objects"

        oracle = pc.mode(pa.array(values, type=pa.float64())).to_pylist()[0]["mode"]
        assert math.isnan(oracle), f"sanity check on the oracle itself failed: {oracle!r}"

        result = reduce_agg("mode", values)
        assert result is not None and math.isnan(result), f"mode={result!r} != PyArrow oracle mode={oracle!r} (nan)"


class TestReduceAggMedianDoesNotSkipNanDocumentedDivergence:
    def test_median_of_value_and_nan_returns_nan(self) -> None:
        """Documented divergence: median does not skip NaN like pandas' skipna median."""
        result = reduce_agg("median", [1.0, float("nan")])
        assert result is not None and math.isnan(result)


class TestGroupKeyValueMergesSignedZeroDocumentedDivergence:
    def test_positive_and_negative_zero_collide_into_one_group(self) -> None:
        """Documented divergence: group_key_value merges 0.0 and -0.0 into one group."""
        groups: dict[Any, list[int]] = {}
        for i, v in enumerate([0.0, -0.0]):
            groups.setdefault(group_key_value(v), []).append(i)
        assert len(groups) == 1
        assert groups[group_key_value(0.0)] == [0, 1]
