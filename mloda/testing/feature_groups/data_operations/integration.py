"""Reusable integration test base for data-operations feature groups.

Subclasses fill the abstract methods and inherit the pipeline, discovery and pattern-matching tests.
Operation-group checks live in OperationGroupIntegrationTestMixin (mixins/operation_group_integration.py).
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import pytest
from mloda.provider import FeatureGroup, OperationFeatureGroup
from mloda.user import DataAccessCollection, Feature, FeatureResolutionError, Options, PluginCollector, mloda

from mloda.testing.data_creator.base import column_selective
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    extract_column,
    result_column_names,
)


def load_group_implementations(group: Any) -> None:
    """Import the manifest next to an operation group, which loads every installed Implementation of it.

    A package without a manifest module is left alone; any other import error is raised.
    """
    if isinstance(group, type) and issubclass(group, OperationFeatureGroup):
        manifest = group.__module__.rsplit(".", 1)[0] + ".manifest"
        try:
            importlib.import_module(manifest)
        except ModuleNotFoundError as exc:
            if exc.name != manifest:
                raise


def enabled_with_implementations(classes: set[Any]) -> PluginCollector:
    """``PluginCollector.enabled_feature_groups(classes)`` after loading the Implementations of any operation group."""
    for cls in classes:
        load_group_implementations(cls)
    return PluginCollector.enabled_feature_groups(classes)


class DataOpsIntegrationTestBase(ABC):
    """Abstract base class for data-ops integration tests through mloda pipeline.

    Subclasses implement the abstract methods below. Override ``expected_row_count`` for reducing operations,
    set ``compare_sorted`` where result order is not guaranteed, ``use_approx`` for floats, and ``nan_is_null``
    if NaN stands for null.
    """

    # -- Abstract methods: subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def feature_group_class(cls) -> Any:
        """Return the feature group implementation class (e.g. WindowAggregationFeatureGroup)."""

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
    def missing_config_feature_name(cls) -> str:
        """Feature name that must not match without config; override when the primary name matches by pattern alone."""
        return cls.primary_feature_name()

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
    def nan_is_null(cls) -> bool:
        """If True, a NaN in the result matches an expected None."""
        return False

    @classmethod
    def approx_rel(cls) -> float:
        """Relative tolerance for pytest.approx. Default: 1e-3."""
        return 1e-3

    @classmethod
    def pipeline_compute_frameworks(cls) -> list[type]:
        """Compute frameworks enabled for the pipeline runs. Default: only ``compute_framework_class()``."""
        return [cls.compute_framework_class()]

    @classmethod
    def group_options(cls) -> dict[str, Any]:
        """Group options set on every feature built by the kit (e.g. ``allow_variants``). Default: none."""
        return {}

    @classmethod
    def data_access_collection(cls) -> DataAccessCollection | None:
        """Data access passed to every kit run_all, e.g. credentials a handle in ``group_options()`` names. Default: none."""
        return None

    # -- Helpers --------------------------------------------------------------

    def _make_feature(self, name: str, options_context: dict[str, Any]) -> Feature:
        """Create a Feature with the given name, group options and options context."""
        return Feature(name, options=Options(group=self.group_options(), context=options_context))

    def _run_all(self, feature: Feature, **kwargs: Any) -> Any:
        """``mloda.run_all`` for one feature with the ``data_access_collection()``."""
        return mloda.run_all([feature], data_access_collection=self.data_access_collection(), **kwargs)

    def _plugin_collector(self, creator: type | None = None) -> PluginCollector:
        """Create a PluginCollector with the feature group and data creator (or ``creator``) enabled."""
        return enabled_with_implementations({creator or self.data_creator_class(), self.feature_group_class()})

    def _run_single_feature(self, name: str, options_context: dict[str, Any], creator: type | None = None) -> Any:
        """Run a single feature through the pipeline and return the framework result holding it."""
        feature = self._make_feature(name, options_context)
        results = self._run_all(
            feature,
            compute_frameworks=self.pipeline_compute_frameworks(),
            plugin_collector=self._plugin_collector(creator) if creator is not None else self._plugin_collector(),
            output_framework=self.compute_framework_class(),
        )
        assert len(results) >= 1

        result_table = next((t for t in results if name in result_column_names(t)), None)
        assert result_table is not None, f"No result table with {name} found"
        return result_table

    def _assert_values_equal(self, actual: list[Any], expected: list[Any]) -> None:
        """Compare actual and expected values, respecting compare_sorted and use_approx."""
        if self.compare_sorted():
            actual = sorted(actual, key=lambda x: (x is None, x))
            expected = sorted(expected, key=lambda x: (x is None, x))

        assert_values_with_nulls(
            actual, expected, approx=self.use_approx(), rel=self.approx_rel(), nan_is_null=self.nan_is_null()
        )

    # -- Concrete test methods (inherited for free) ---------------------------

    def _assert_pipeline(
        self, name: str, options: dict[str, Any], expected: list[Any], creator: type | None = None
    ) -> None:
        """Run the feature, then check the row count and values."""
        result_col = extract_column(self._run_single_feature(name, options, creator), name)
        assert len(result_col) == self.expected_row_count()
        self._assert_values_equal(result_col, expected)

    def test_primary_feature_through_pipeline(self) -> None:
        """Run the primary feature through run_all and verify values."""
        self._assert_pipeline(
            self.primary_feature_name(), self.primary_feature_options(), self.primary_expected_values()
        )

    def test_primary_feature_with_column_selective_source(self) -> None:
        """Key columns named in the options must be requested from a source that serves only requested columns."""
        self._assert_pipeline(
            self.primary_feature_name(),
            self.primary_feature_options(),
            self.primary_expected_values(),
            column_selective(self.data_creator_class()),
        )

    def test_secondary_feature_through_pipeline(self) -> None:
        """Run the secondary feature through run_all and verify values."""
        self._assert_pipeline(
            self.secondary_feature_name(), self.secondary_feature_options(), self.secondary_expected_values()
        )

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
            self._run_all(
                feature,
                compute_frameworks=self.pipeline_compute_frameworks(),
                plugin_collector=plugin_collector,
                output_framework=self.compute_framework_class(),
            )

    def test_family_base_is_not_selected_without_an_implementation(self) -> None:
        """With only the family base(s) and the source enabled, resolution fails as abstract-only."""
        if issubclass(self.feature_group_class(), OperationFeatureGroup):
            pytest.skip("an OperationFeatureGroup is concrete; the rejection test covers a missing framework")
        bases = [
            base
            for base in self.feature_group_class().__mro__[1:]
            if isinstance(base, type)
            and issubclass(base, FeatureGroup)
            and base is not FeatureGroup
            and inspect.isabstract(base)
            and "PREFIX_PATTERN" in vars(base)
        ]
        feature = self._make_feature(self.primary_feature_name(), self.primary_feature_options())
        # A subclass may override PREFIX_PATTERN, so keep only bases that match the primary name.
        bases = [
            base
            for base in bases
            if base.match_feature_group_criteria(self.primary_feature_name(), feature.options, None)
        ]
        if not bases:
            pytest.skip("no data-operations family base matches the primary feature")

        plugin_collector = PluginCollector.enabled_feature_groups({self.data_creator_class(), *bases})

        with pytest.raises(FeatureResolutionError) as excinfo:
            self._run_all(
                feature,
                compute_frameworks=[self.compute_framework_class()],
                plugin_collector=plugin_collector,
            )
        assert excinfo.value.result.failure_kind == "abstract_only"

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
        """Verify match_feature_group_criteria rejects ``missing_config_feature_name()`` without config."""
        options = Options()
        assert not self.feature_group_class().match_feature_group_criteria(self.missing_config_feature_name(), options)
