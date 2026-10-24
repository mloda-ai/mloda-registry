"""Reusable operation-group integration test mixin for data-operations feature groups.

Provides 2 integration test methods that verify an OperationFeatureGroup is served
by one Implementation per compute framework through mloda's full pipeline.

An integration test class mixes this in alongside DataOpsIntegrationTestBase; no
configuration beyond the base's is needed.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import Implementation, get_all_subclasses
from mloda.user import FeatureResolutionError, PluginCollector, mloda


class OperationGroupIntegrationTestMixin:
    """Mixin providing operation-group integration tests through mloda's full pipeline.

    Requires the host class to provide (from DataOpsIntegrationTestBase):
    - ``feature_group_class()``
    - ``compute_framework_class()``
    - ``primary_feature_name()`` / ``primary_feature_options()`` / ``primary_expected_values()``
    - ``_make_feature(name, options_context)``
    - ``_plugin_collector()``
    - ``_assert_values_equal(actual, expected)``
    """

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

    def test_several_frameworks_enabled_runs(self) -> None:
        """Enabling a second framework still plans one compute step for the group and returns the same values."""
        pytest.importorskip("pandas")
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        group = self.feature_group_class()  # type: ignore[attr-defined]
        frameworks = [self.compute_framework_class(), PandasDataFrame]  # type: ignore[attr-defined]
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
        tables = [t for t in results if isinstance(t, pa.Table) and name in t.column_names]
        assert len(tables) == 1
        self._assert_values_equal(  # type: ignore[attr-defined]
            tables[0].column(name).to_pylist(),
            self.primary_expected_values(),  # type: ignore[attr-defined]
        )
