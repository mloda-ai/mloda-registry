"""Reusable integration test base for data-operations feature groups.

Provides concrete test methods that verify feature groups work end-to-end
through mloda's pipeline (run_all, PluginCollector, pattern matching).

Subclasses implement a few abstract methods to wire up their specific
feature group, then inherit 7 concrete test methods for free:

- 2 single-feature pipeline tests (primary + secondary)
- 2 plugin discovery tests (discoverable + blocked when disabled)
- 3 pattern matching tests (valid names, invalid names, missing config)

Usage::

    class TestWindowAggregationIntegration(DataOpsIntegrationTestBase):

        @classmethod
        def feature_group_class(cls):
            return PyArrowWindowAggregation

        @classmethod
        def data_creator_class(cls):
            return PyArrowDataOpsTestDataCreator

        # ... (see abstract methods below)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature, PluginCollector, mloda


class DataOpsIntegrationTestBase(ABC):
    """Abstract base class for data-ops integration tests through mloda pipeline.

    Subclasses must implement the abstract methods below. The base class
    provides 7 concrete test methods that exercise the full mloda pipeline
    including run_all, PluginCollector, and pattern matching.

    For row-preserving operations (window_aggregation, scalar_aggregate, rank, offset),
    the expected row count is 12 (same as input). For reducing operations
    (aggregation), override ``expected_row_count`` to return the reduced count.

    For operations where result order is not guaranteed (aggregation),
    set ``compare_sorted`` to True.

    For floating-point results, set ``use_approx`` to True.
    """

    # -- Abstract methods: subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def feature_group_class(cls) -> Any:
        """Return the feature group implementation class (e.g. PyArrowWindowAggregation)."""

    @classmethod
    @abstractmethod
    def data_creator_class(cls) -> type:
        """Return the data creator class (e.g. PyArrowDataOpsTestDataCreator)."""

    @classmethod
    @abstractmethod
    def compute_framework_class(cls) -> type:
        """Return the compute framework class (e.g. PyArrowTable)."""

    @classmethod
    @abstractmethod
    def primary_feature_name(cls) -> str:
        """Feature name for the primary single-feature test (e.g. 'value_int__sum_window')."""

    @classmethod
    @abstractmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        """Options context dict for the primary feature (e.g. {'partition_by': ['region']})."""

    @classmethod
    @abstractmethod
    def primary_expected_values(cls) -> list[Any]:
        """Expected column values for the primary feature."""

    @classmethod
    @abstractmethod
    def secondary_feature_name(cls) -> str:
        """Feature name for the secondary single-feature test (e.g. 'value_int__avg_window')."""

    @classmethod
    @abstractmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        """Options context dict for the secondary feature."""

    @classmethod
    @abstractmethod
    def secondary_expected_values(cls) -> list[Any]:
        """Expected column values for the secondary feature."""

    @classmethod
    @abstractmethod
    def valid_feature_names(cls) -> list[str]:
        """Feature names that should match this feature group (for pattern matching tests)."""

    @classmethod
    @abstractmethod
    def invalid_feature_names(cls) -> list[str]:
        """Feature names that should NOT match this feature group."""

    @classmethod
    @abstractmethod
    def match_options(cls) -> Options:
        """Options instance for pattern matching tests (valid config)."""

    # -- Overridable configuration -------------------------------------------

    @classmethod
    def expected_row_count(cls) -> int:
        """Expected number of rows in results. Default: 12 (row-preserving)."""
        return 12

    @classmethod
    def compare_sorted(cls) -> bool:
        """If True, sort values before comparing. Use for reducing operations."""
        return False

    @classmethod
    def use_approx(cls) -> bool:
        """If True, use pytest.approx for value comparison."""
        return False

    @classmethod
    def approx_rel(cls) -> float:
        """Relative tolerance for pytest.approx. Default: 1e-3."""
        return 1e-3

    # -- Helpers --------------------------------------------------------------

    def _make_feature(self, name: str, options_context: dict[str, Any]) -> Feature:
        """Create a Feature with the given name and options context."""
        return Feature(name, options=Options(context=options_context))

    def _plugin_collector(self) -> PluginCollector:
        """Create a PluginCollector with the feature group and data creator enabled."""
        return PluginCollector.enabled_feature_groups({self.data_creator_class(), self.feature_group_class()})

    def _run_single_feature(self, name: str, options_context: dict[str, Any]) -> pa.Table:
        """Run a single feature through the pipeline and return the result table."""
        feature = self._make_feature(name, options_context)
        results = mloda.run_all(
            [feature],
            compute_frameworks={self.compute_framework_class()},
            plugin_collector=self._plugin_collector(),
        )
        assert len(results) >= 1

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and name in table.column_names:
                result_table = table
                break

        assert result_table is not None, f"No result table with {name} found"
        return result_table

    def _assert_values_equal(self, actual: list[Any], expected: list[Any]) -> None:
        """Compare actual and expected values, respecting compare_sorted and use_approx."""
        if self.compare_sorted():
            actual = sorted(actual, key=lambda x: (x is None, x))
            expected = sorted(expected, key=lambda x: (x is None, x))

        if self.use_approx():
            assert actual == pytest.approx(expected, rel=self.approx_rel())
        else:
            assert actual == expected

    # -- Concrete test methods (inherited for free) ---------------------------

    def test_primary_feature_through_pipeline(self) -> None:
        """Run the primary feature through run_all and verify values."""
        result_table = self._run_single_feature(
            self.primary_feature_name(),
            self.primary_feature_options(),
        )
        assert result_table.num_rows == self.expected_row_count()

        result_col = result_table.column(self.primary_feature_name()).to_pylist()
        self._assert_values_equal(result_col, self.primary_expected_values())

    def test_secondary_feature_through_pipeline(self) -> None:
        """Run the secondary feature through run_all and verify values."""
        result_table = self._run_single_feature(
            self.secondary_feature_name(),
            self.secondary_feature_options(),
        )
        assert result_table.num_rows == self.expected_row_count()

        result_col = result_table.column(self.secondary_feature_name()).to_pylist()
        self._assert_values_equal(result_col, self.secondary_expected_values())

    def test_feature_group_is_discoverable(self) -> None:
        """Verify the feature group can be enabled via PluginCollector."""
        plugin_collector = self._plugin_collector()
        assert plugin_collector.applicable_feature_group_class(self.feature_group_class())
        assert plugin_collector.applicable_feature_group_class(self.data_creator_class())

    def test_disabled_feature_group_blocks_execution(self) -> None:
        """When the feature group is not enabled, pipeline should fail."""
        plugin_collector = PluginCollector.enabled_feature_groups({self.data_creator_class()})

        feature = self._make_feature(
            self.primary_feature_name(),
            self.primary_feature_options(),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={self.compute_framework_class()},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria_valid(self) -> None:
        """Verify match_feature_group_criteria accepts valid feature names."""
        options = self.match_options()
        for name in self.valid_feature_names():
            assert self.feature_group_class().match_feature_group_criteria(name, options), (
                f"Expected {name} to match {self.feature_group_class().__name__}"
            )

    def test_match_rejects_invalid_feature_names(self) -> None:
        """Verify match_feature_group_criteria rejects non-matching feature names."""
        options = self.match_options()
        for name in self.invalid_feature_names():
            assert not self.feature_group_class().match_feature_group_criteria(name, options), (
                f"Expected {name} to NOT match {self.feature_group_class().__name__}"
            )

    def test_match_rejects_missing_config(self) -> None:
        """Verify match_feature_group_criteria rejects when required config is missing."""
        options = Options()
        assert not self.feature_group_class().match_feature_group_criteria(self.primary_feature_name(), options)
