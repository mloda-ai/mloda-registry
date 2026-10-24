"""Reusable operation-group integration test mixin for data-operations feature groups.

Verifies an OperationFeatureGroup is served by one Implementation per compute framework:
rejected when none of its frameworks is enabled, resolution error without an Implementation on the
framework, and runnable with several frameworks enabled. Divergence routing and resolve-time
rejection read an overridable hook and skip when it is empty.
Mix into an integration test class alongside DataOpsIntegrationTestBase.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from mloda.provider import Implementation, get_all_subclasses
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda

from mloda.testing.feature_groups.data_operations.helpers import extract_column, result_column_names

PYARROW_TABLE = "mloda_plugins.compute_framework.base_implementations.pyarrow.table:PyArrowTable"
PANDAS_DATAFRAME = "mloda_plugins.compute_framework.base_implementations.pandas.dataframe:PandasDataFrame"
SQLITE_FRAMEWORK = "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework:SqliteFramework"


@dataclass(frozen=True)
class RoutingCase:
    """A feature and the (implementation, compute framework) pair that must serve it.

    ``implementation`` and ``compute_framework`` are ``"module:Name"`` strings, resolved lazily so a case
    can name an optional backend.
    """

    feature_name: str
    options: dict[str, Any] = field(default_factory=dict)
    implementation: str = ""
    compute_framework: str = ""


def _resolve(qualified: str) -> Any:
    module, name = qualified.split(":")
    return getattr(importlib.import_module(module), name)


class OperationGroupIntegrationTestMixin:
    """Mixin providing operation-group integration tests through mloda's full pipeline.

    Requires the host class to provide (from DataOpsIntegrationTestBase):
    - ``feature_group_class()``
    - ``compute_framework_class()`` / ``pipeline_compute_frameworks()``
    - ``primary_feature_name()`` / ``primary_feature_options()`` / ``primary_expected_values()``
    - ``_make_feature(name, options_context)``
    - ``_plugin_collector()``
    - ``_assert_values_equal(actual, expected)``

    Optional hooks:
    - ``several_frameworks()``: the frameworks enabled together in ``test_several_frameworks_enabled_runs``.
    - ``divergent_routing_cases()``: features that data held in the ``routing_data_creator()`` framework
      must see served by another Implementation (a divergent one) or kept on it (a non-divergent one).
    - ``routing_data_creator()``: the data creator whose framework holds the routed data. Default: Pandas.
    - ``unsupported_framework_case()``: a (feature name, options) that ``compute_framework_class()`` alone
      cannot serve, so a run on it is rejected at resolve time.
    """

    @classmethod
    def divergent_routing_cases(cls) -> list[RoutingCase]:
        """Cases for ``test_data_is_routed_by_divergence``. Default: none (the test skips)."""
        return []

    @classmethod
    def routing_data_creator(cls) -> type[Any]:
        """The data creator for ``test_data_is_routed_by_divergence``. Default: Pandas (skips without it)."""
        pytest.importorskip("pandas")
        from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator

        return PandasDataOpsTestDataCreator

    @classmethod
    def unsupported_framework_case(cls) -> tuple[str, dict[str, Any]] | None:
        """A (feature name, options) rejected on ``compute_framework_class()``. Default: none (skips)."""
        return None

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

        return [*cls.pipeline_compute_frameworks(), PandasDataFrame]  # type: ignore[attr-defined]

    def test_framework_without_implementation_is_a_resolution_error(self) -> None:
        """Without an Implementation of the group on the framework, the framework cannot serve the feature.

        A group that ships none there is run as is; otherwise every Implementation there is disabled.
        """
        # Building the collector loads the group's Implementations, so it comes first.
        collector = self._plugin_collector()  # type: ignore[attr-defined]
        group = self.feature_group_class()  # type: ignore[attr-defined]
        assert any(getattr(impl, "feature_group", None) is group for impl in get_all_subclasses(Implementation))
        implementations = self._implementations_on_framework()
        if implementations:
            collector.add_disabled_implementation_classes(implementations)
        feature = self._make_feature(  # type: ignore[attr-defined]
            self.primary_feature_name(),  # type: ignore[attr-defined]
            self.primary_feature_options(),  # type: ignore[attr-defined]
        )

        with pytest.raises(FeatureResolutionError) as excinfo:
            mloda.run_all(
                [feature],
                compute_frameworks=[self.compute_framework_class()],  # type: ignore[attr-defined]
                plugin_collector=collector,
            )
        message = str(excinfo.value)
        assert group.__name__ in message
        if implementations:
            assert "disabled or denied" in message

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

        results = mloda.run_all(
            [feature],
            compute_frameworks=frameworks,
            plugin_collector=collector,
            output_framework=self.compute_framework_class(),  # type: ignore[attr-defined]
        )
        name = self.primary_feature_name()  # type: ignore[attr-defined]
        tables = [t for t in results if name in result_column_names(t)]
        assert len(tables) == 1
        self._assert_values_equal(  # type: ignore[attr-defined]
            extract_column(tables[0], name),
            self.primary_expected_values(),  # type: ignore[attr-defined]
        )

    def test_data_is_routed_by_divergence(self) -> None:
        """Data already in the creator's framework is served by the declared Implementation, not always by it."""
        cases = self.divergent_routing_cases()
        if not cases:
            pytest.skip("the group declares no divergent routing cases")
        from mloda.testing.feature_groups.data_operations.integration import enabled_with_implementations

        creator = self.routing_data_creator()
        group = self.feature_group_class()  # type: ignore[attr-defined]
        frameworks = list(dict.fromkeys([creator.compute_framework, *self.pipeline_compute_frameworks()]))  # type: ignore[attr-defined]
        collector = enabled_with_implementations({creator, group})
        for case in cases:
            feature = Feature(case.feature_name, options=Options(context=case.options))
            plan = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=collector)
            steps = [step for step in plan if step.step_kind == "compute" and step.feature_group is group]
            assert len(steps) == 1, case.feature_name
            assert steps[0].implementation is _resolve(case.implementation), case.feature_name
            assert steps[0].compute_framework is _resolve(case.compute_framework), case.feature_name

    def test_unsupported_input_is_rejected_at_resolve_time(self) -> None:
        """An input the framework's Implementation does not support is dropped at match time, not at compute time."""
        case = self.unsupported_framework_case()
        if case is None:
            pytest.skip("every input is supported on the compute framework under test")
        name, options = case
        feature = self._make_feature(name, options)  # type: ignore[attr-defined]
        with pytest.raises(FeatureResolutionError, match="supports_compute_framework rejected"):
            mloda.run_all(
                [feature],
                compute_frameworks=[self.compute_framework_class()],  # type: ignore[attr-defined]
                plugin_collector=self._plugin_collector(),  # type: ignore[attr-defined]
            )
