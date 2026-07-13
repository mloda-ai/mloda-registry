"""Tests for CommunityExampleExtender."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.extenders.example import CommunityExampleExtender

# Canonical value_int column of the shared 12-row test dataset.
_VALUE_INT = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]


class FailingCommunityExampleExtender(CommunityExampleExtender):
    """Deliberately failing extender: its own code raises before delegating to func."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("extender boom")


class CountingCommunityExampleExtender(CommunityExampleExtender):
    """Healthy pass-through extender that records how often it was invoked."""

    def __init__(self, raise_on_error: bool = True) -> None:
        super().__init__(raise_on_error=raise_on_error)
        self.calls = 0

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return super().__call__(func, *args, **kwargs)


class FailingCalculateFeatureGroup(FeatureGroup):
    """Root feature group whose calculate_feature always raises, counting its invocations."""

    calls = 0

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"boom_feature"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        raise RuntimeError("inner boom")


def _run_value_int(*extenders: Extender) -> list[Any]:
    """Run the minimal ``value_int`` feature through run_all with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

    results = mloda.run_all(
        ["value_int"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )

    for table in results:
        if "value_int" in table.column_names:
            values: list[Any] = table.to_pydict()["value_int"]
            return values

    raise AssertionError("No result table with value_int found")


def _run_boom_feature(*extenders: Extender) -> Any:
    """Run the always-failing ``boom_feature`` through run_all with the given extenders."""
    plugin_collector = PluginCollector.enabled_feature_groups({FailingCalculateFeatureGroup})

    return mloda.run_all(
        ["boom_feature"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender=set(extenders),
    )


class TestCommunityExampleExtenderImport:
    """Test that CommunityExampleExtender can be imported."""

    def test_import_from_package(self) -> None:
        """Test that CommunityExampleExtender can be imported from the package."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert CommunityExampleExtender is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert isinstance(CommunityExampleExtender, type)


class TestCommunityExampleExtenderInheritance:
    """Test that CommunityExampleExtender inherits from Extender."""

    def test_inherits_from_extender(self) -> None:
        """Test that CommunityExampleExtender is a subclass of Extender."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert issubclass(CommunityExampleExtender, Extender)

    def test_instance_is_extender(self) -> None:
        """Test that an instance is an instance of Extender."""
        from mloda.community.extenders.example import CommunityExampleExtender

        instance = CommunityExampleExtender()
        assert isinstance(instance, Extender)


class TestCommunityExampleExtenderBasicFunctionality:
    """Test basic functionality of CommunityExampleExtender."""

    def test_has_name_attribute(self) -> None:
        """Test that the extender has a name or identifier."""
        from mloda.community.extenders.example import CommunityExampleExtender

        # Extender implementations should have some form of identification
        instance = CommunityExampleExtender()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "CommunityExampleExtender"

    def test_can_instantiate(self) -> None:
        """Test that the extender can be instantiated."""
        from mloda.community.extenders.example import CommunityExampleExtender

        instance = CommunityExampleExtender()
        assert instance is not None


class TestCommunityExampleExtenderErrorContract:
    """raise_on_error, wraps() and the pass-through __call__."""

    def test_raise_on_error_defaults_to_true(self) -> None:
        """Default is breaking: a failure propagates."""
        assert CommunityExampleExtender().raise_on_error is True

    def test_raise_on_error_can_be_disabled(self) -> None:
        """raise_on_error=False marks the extender as warning-only."""
        assert CommunityExampleExtender(raise_on_error=False).raise_on_error is False

    def test_wraps_calculate_feature_hook(self) -> None:
        """The extender must hook feature calculation, otherwise it never runs."""
        assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in CommunityExampleExtender().wraps()

    def test_call_is_pass_through(self) -> None:
        """__call__ forwards args and returns the wrapped function's result."""
        assert CommunityExampleExtender()(lambda x, y: x + y, 1, 2) == 3


class TestCommunityExampleExtenderRunAll:
    """End-to-end raise_on_error semantics through mloda.run_all."""

    def test_pass_through_extender_does_not_change_result(self) -> None:
        """The example extender is a no-op: the feature computes normally."""
        assert _run_value_int(CommunityExampleExtender()) == _VALUE_INT

    def test_failing_extender_breaks_run_by_default(self) -> None:
        """raise_on_error=True (default): the extender failure propagates out of run_all."""
        with pytest.raises(Exception, match="extender boom"):
            _run_value_int(FailingCommunityExampleExtender())

    def test_failing_extender_warns_only_when_raise_on_error_false(self, caplog: pytest.LogCaptureFixture) -> None:
        """raise_on_error=False: failure is logged as a warning and the feature still computes."""
        with caplog.at_level(logging.WARNING):
            values = _run_value_int(FailingCommunityExampleExtender(raise_on_error=False))

        assert values == _VALUE_INT

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("extender boom" in message for message in warnings), warnings

    def test_warning_only_failure_does_not_stop_other_extender_in_chain(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chained extenders: a warning-only failure is contained, the healthy one still runs."""
        healthy = CountingCommunityExampleExtender()

        with caplog.at_level(logging.WARNING):
            values = _run_value_int(FailingCommunityExampleExtender(raise_on_error=False), healthy)

        assert values == _VALUE_INT
        assert healthy.calls > 0

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("extender boom" in message for message in warnings), warnings

    def test_wrapped_function_failure_propagates_and_runs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failure of the wrapped function propagates even for a warning-only extender, without a re-run."""
        FailingCalculateFeatureGroup.calls = 0

        with caplog.at_level(logging.WARNING):
            with pytest.raises(Exception, match="inner boom"):
                _run_boom_feature(CommunityExampleExtender(raise_on_error=False))

        assert FailingCalculateFeatureGroup.calls == 1

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("inner boom" in message for message in warnings), warnings
