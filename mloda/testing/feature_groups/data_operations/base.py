"""Shared abstract base class for all data-operations framework test suites.

``DataOpsTestBase`` eliminates the boilerplate that was duplicated across all
9 operation-specific test bases (scalar_aggregate, aggregation, window_aggregation,
offset, rank, binning, datetime, frame_aggregate, string). It provides:

- 5 abstract adapter methods (implementation_class, create_test_data,
  extract_column, get_row_count, get_expected_type) that framework mixins
  or concrete test classes implement once.
- ``source_arrow_table``: overridable hook for the fixture setup_method builds from.
- ``contract_version``: hook plus an inherited check against the group's ``CONTRACT_VERSION``.
- Shared setup_method / teardown_method (connection lifecycle).
- A ``_skip_if_unsupported`` helper for operations not available on every framework.
- A ``_compare_with_reference`` cross-framework comparison helper.

Operation-specific test bases (e.g. ``AggregationTestBase``) inherit from this
class and add their own expected values and concrete test methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import DivergenceKind, Implementation

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import (
    calculate_with,
    feature_group_of,
    is_null,
    result_column_names,
)
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column

# Capability methods ``_skip_if_unsupported`` probes, in order. Operation families
# spell the same idea differently: aggregation has "agg_types", binning "ops",
# offset "offset_types", rank "rank_types".
_SUPPORTED_SET_ATTRS: tuple[str, ...] = (
    "supported_agg_types",
    "supported_ops",
    "supported_offset_types",
    "supported_rank_types",
)


def divergent_subtypes(implementation: Any) -> frozenset[str]:
    """Subtypes the Implementation's non-GAP ``divergences`` records name; a scopeless record covers the universe."""
    covered: set[str] = set()
    for record in implementation.divergences:
        if record.kind is DivergenceKind.GAP:
            continue
        if record.subtypes:
            covered |= record.subtypes
        elif not record.outputs:
            covered |= implementation.feature_group.subtype_universe()
    return frozenset(covered)


class DataOpsTestBase(ABC):
    """Abstract base class shared by all data-operations test suites.

    Subclasses must implement the 5 adapter methods below (or inherit them
    from a framework mixin such as ``PandasTestMixin``).

    Operation-specific bases extend this with concrete test methods and
    expected-value constants.
    """

    # -- Abstract adapter methods (implement once per framework) ----------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the feature-group implementation class under test."""

    @abstractmethod
    def create_test_data(self, arrow_table: pa.Table) -> Any:
        """Convert the canonical PyArrow test table to the framework's native format."""

    @abstractmethod
    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        """Extract a column from the framework result as a Python list."""

    @abstractmethod
    def get_row_count(self, result: Any) -> int:
        """Return the number of rows in the framework result."""

    @abstractmethod
    def get_expected_type(self) -> Any:
        """Return the expected Python type of the framework result."""

    # -- Reference implementation (for cross-framework comparison) --------------------

    @classmethod
    def reference_implementation_class(cls) -> Any:
        """Return the reference implementation for cross-framework comparison.

        Operation-specific bases override this to import the correct class.
        """
        raise NotImplementedError("Subclass must override reference_implementation_class")

    @classmethod
    def reference_calculate(cls, table: pa.Table, features: Any) -> Any:
        """Run the reference (a FeatureGroup or Implementation class) on the canonical PyArrow table; override for a function."""
        return calculate_with(cls.reference_implementation_class(), table, features)

    @classmethod
    def contract_version(cls) -> int:
        """Group contract version this base's expected values were written for; override when a family's tables move."""
        return 1

    def test_group_contract_version_is_the_kit_version(self) -> None:
        group = feature_group_of(self.implementation_class())
        actual = getattr(group, "CONTRACT_VERSION", None)
        assert actual == self.contract_version(), (
            f"{group.__name__}.CONTRACT_VERSION is {actual}, but the kit expects {self.contract_version()}; "
            "override contract_version() once the expected values follow the new contract"
        )

    # -- Setup / teardown ------------------------------------------------------

    @classmethod
    def source_arrow_table(cls) -> pa.Table:
        """PyArrow table setup_method builds ``self._arrow_table`` / ``self.test_data`` from.

        Default: canonical 12-row dataset, override for a dedicated fixture.
        """
        return PyArrowDataOpsTestDataCreator.create()

    def setup_method(self) -> None:
        """Create test data from ``source_arrow_table()``.

        Connection-based mixins (DuckdbTestMixin, SqliteTestMixin) create
        ``self.conn`` first, then call ``super().setup_method()``.
        """
        self._arrow_table = self.source_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        """Close ``self.conn`` if set by a connection-based mixin."""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    # -- Helpers ---------------------------------------------------------------

    def _skip_if_unsupported(self, op: str) -> None:
        """Skip a test if *op* is not in the framework's supported set.

        Works with any subclass that defines ``supported_agg_types``,
        ``supported_ops``, ``supported_offset_types``, or ``supported_rank_types``.
        A subclass defining none of them is a defect, not an unsupported op, so
        this raises rather than skipping: skipping would make the shared test
        pass on every framework without ever running.
        """
        for attr in _SUPPORTED_SET_ATTRS:
            method = getattr(self, attr, None)
            if method is not None:
                if op not in method():
                    pytest.skip(f"{op} not supported by this framework")
                return
        raise TypeError(
            f"{type(self).__name__} declares none of {', '.join(_SUPPORTED_SET_ATTRS)}, "
            f"so {op!r} cannot be checked for support. Declare the supported set on the test base, "
            "or keep the shared test on a base that declares one."
        )

    def _snapshot(self, data: Any) -> dict[str, list[Any]]:
        """Deep copy of ``data`` as column name -> list of values, via ``extract_column``."""
        return {name: list(self.extract_column(data, name)) for name in result_column_names(data)}

    def _assert_snapshot_unchanged(self, before: dict[str, list[Any]], data: Any) -> None:
        """Assert ``data`` still holds the ``before`` snapshot (NaN equal to NaN, None equal to None)."""
        after = self._snapshot(data)
        assert list(after) == list(before), f"input columns changed: {list(before)} -> {list(after)}"
        for name, expected in before.items():
            actual = after[name]
            assert len(actual) == len(expected), f"input column {name} row count changed"
            for i, (a, e) in enumerate(zip(actual, expected)):
                same = (is_null(a) and is_null(e) and (a is None) == (e is None)) or a == e
                assert same, f"input column {name} row {i} mutated: {e!r} -> {a!r}"

    # -- Shared NaN-policy fixture -----------------------------------------

    @classmethod
    def nan_policy_table(cls) -> pa.Table:
        """Shared grp/ts/val fixture for sum/avg/count/median/mode/min/max/std/var NaN-policy tests.

        Consumed by ``NanPolicyTestMixin.test_mixin_nan_policy`` (see
        ``mloda/testing/feature_groups/data_operations/mixins/nan_policy.py``).
        val mixes NaN with real values; no grp or window is all-NaN, but resampling into
        1-day buckets does produce all-NaN buckets on purpose.
        """
        return pa.table(
            {
                "grp": ["A", "A", "A", "B", "B", "B", "B"],
                "ts": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 3, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 3, tzinfo=timezone.utc),
                    datetime(2024, 1, 4, tzinfo=timezone.utc),
                ],
                "val": pa.array([2.0, float("nan"), 1.0, 1.0, float("nan"), 3.0, float("nan")], type=pa.float64()),
            }
        )

    @classmethod
    def nan_divergent_agg_types(cls) -> frozenset[str]:
        """Agg types where this framework pins a known NaN divergence instead of the policy.

        Default: the subtypes its ``divergences`` name for the Implementation under test, empty for any other plugin.
        Read by ``NanPolicyTestMixin.test_mixin_nan_policy``; see docs/guides/data-operation-patterns/known-divergences.md.
        """
        impl = cls.implementation_class()
        if isinstance(impl, type) and issubclass(impl, Implementation):
            return divergent_subtypes(impl)
        return frozenset()

    # -- Cross-framework comparison --------------------------------------------

    def _compare_with_reference(
        self,
        feature_name: str,
        *,
        partition_by: list[str] | None = None,
        order_by: str | None = None,
        use_approx: bool = False,
        rel: float = 1e-6,
        passthrough_cols: tuple[str, ...] = (),
        data: Any = None,
        **extra_context: Any,
    ) -> None:
        """Run a feature through this framework and the reference, assert results match.

        ``passthrough_cols`` are input columns that must also equal the reference's.
        ``data`` replaces ``self.test_data`` for the framework run (the reference still uses ``_arrow_table``).

        ``extra_context`` is forwarded to ``make_feature_set`` as op-specific Options context.
        """
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        fs = make_feature_set(feature_name, partition_by=partition_by, order_by=order_by, **extra_context)
        result = calculate_with(self.implementation_class(), self.test_data if data is None else data, fs)
        ref = self.reference_calculate(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"

        for c in passthrough_cols:
            assert self.extract_column(result, c) == _extract_column(ref, c), f"passthrough column {c} differs"

        if use_approx:
            for i, (ref_val, fw_val) in enumerate(zip(ref_col, result_col)):
                if ref_val is None:
                    assert fw_val is None, f"row {i}: expected None, got {fw_val}"
                else:
                    assert fw_val == pytest.approx(ref_val, rel=rel), f"row {i}: {fw_val} != reference {ref_val}"
        else:
            assert result_col == ref_col
