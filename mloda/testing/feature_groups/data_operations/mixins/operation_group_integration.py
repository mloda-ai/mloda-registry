"""Reusable operation-group integration test mixin for data-operations feature groups.

Verifies an OperationFeatureGroup is served by one Implementation per compute framework:
rejected when none of its frameworks is enabled, resolution error without an Implementation on the
framework, and runnable with several frameworks enabled.
Mix into an integration test class alongside DataOpsIntegrationTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.provider import Implementation, get_all_subclasses
from mloda.user import FeatureResolutionError, PluginCollector, mloda

from mloda.testing.feature_groups.data_operations.helpers import extract_column, result_column_names


class OperationGroupIntegrationTestMixin:
    """Operation-group integration tests; the host class is a DataOpsIntegrationTestBase."""

    @classmethod
    def _implementations_on_framework(cls) -> set[type[Any]]:
        """The Implementations of the group on ``compute_framework_class()``."""
        group = cls.feature_group_class()  # type: ignore[attr-defined]
        framework = cls.compute_framework_class()  # type: ignore[attr-defined]
        return {
            impl
            for impl in get_all_subclasses(Implementation)
            if getattr(impl, "feature_group", None) is group and getattr(impl, "compute_framework", None) is framework
        }

    @classmethod
    def several_frameworks(cls) -> list[type[Any]]:
        """The frameworks enabled together in the several-frameworks run."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        return [cls.compute_framework_class(), PandasDataFrame]  # type: ignore[attr-defined]

    def test_framework_without_implementation_is_a_resolution_error(self) -> None:
        """With every Implementation of the group on the framework disabled, the framework cannot serve the feature."""
        implementations = self._implementations_on_framework()
        assert implementations, "the group has no Implementation on the compute framework under test"

        collector = self._plugin_collector()  # type: ignore[attr-defined]
        collector.add_disabled_implementation_classes(implementations)
        feature = self._make_feature(  # type: ignore[attr-defined]
            self.primary_feature_name(),  # type: ignore[attr-defined]
            self.primary_feature_options(),  # type: ignore[attr-defined]
        )

        with pytest.raises(FeatureResolutionError):
            mloda.run_all(
                [feature],
                compute_frameworks=[self.compute_framework_class()],  # type: ignore[attr-defined]
                plugin_collector=collector,
            )

    def test_group_is_rejected_when_none_of_its_frameworks_is_enabled(self) -> None:
        """An operation group with a loaded Implementation fails on a framework it has no Implementation for."""
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        from mloda.testing.data_creator.polars import PolarsDataOpsTestDataCreator

        group = self.feature_group_class()  # type: ignore[attr-defined]
        frameworks = group.compute_framework_definition()
        assert frameworks, f"{group.__name__} has no loaded Implementation"
        if PolarsDataFrame in frameworks:
            pytest.skip("eager Polars is served by this group")

        plugin_collector = PluginCollector.enabled_feature_groups({PolarsDataOpsTestDataCreator, group})
        feature = self._make_feature(  # type: ignore[attr-defined]
            self.primary_feature_name(),  # type: ignore[attr-defined]
            self.primary_feature_options(),  # type: ignore[attr-defined]
        )
        with pytest.raises(FeatureResolutionError) as excinfo:
            mloda.run_all([feature], compute_frameworks=[PolarsDataFrame], plugin_collector=plugin_collector)
        message = str(excinfo.value)
        assert group.__name__ in message
        assert "none of its compute frameworks are enabled" in message

    def test_several_frameworks_enabled_runs(self) -> None:
        """Enabling a second framework still plans one compute step for the group and returns the same values."""
        group = self.feature_group_class()  # type: ignore[attr-defined]
        frameworks = self.several_frameworks()
        collector: PluginCollector = self._plugin_collector()  # type: ignore[attr-defined]
        feature = self._make_feature(  # type: ignore[attr-defined]
            self.primary_feature_name(),  # type: ignore[attr-defined]
            self.primary_feature_options(),  # type: ignore[attr-defined]
        )

        plan = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=collector)
        steps = [step for step in plan if step.step_kind == "compute" and step.feature_group is group]
        assert len(steps) == 1
        assert steps[0].implementation is not None
        assert steps[0].compute_framework is not None

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=collector)
        name = self.primary_feature_name()  # type: ignore[attr-defined]
        tables = [t for t in results if name in result_column_names(t)]
        assert len(tables) == 1
        self._assert_values_equal(  # type: ignore[attr-defined]
            extract_column(tables[0], name),
            self.primary_expected_values(),  # type: ignore[attr-defined]
        )
