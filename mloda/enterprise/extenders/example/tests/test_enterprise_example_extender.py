"""Tests for EnterpriseExampleExtender."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.steward import Extender, ExtenderHook
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.enterprise.extenders.example import EnterpriseExampleExtender

# Canonical value_int column of the shared 12-row test dataset.
_VALUE_INT = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]


class FailingEnterpriseExampleExtender(EnterpriseExampleExtender):
    """Deliberately failing extender: its own code raises before delegating to func."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("extender boom")


def _run_value_int(extender: Extender) -> list[Any]:
    """Run the minimal ``value_int`` feature through run_all with one extender."""
    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

    results = mloda.run_all(
        ["value_int"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        function_extender={extender},
    )

    for table in results:
        if "value_int" in table.column_names:
            values: list[Any] = table.to_pydict()["value_int"]
            return values

    raise AssertionError("No result table with value_int found")


class TestEnterpriseExampleExtenderImport:
    """Test that EnterpriseExampleExtender can be imported."""

    def test_import_from_package(self) -> None:
        """Test that EnterpriseExampleExtender can be imported from the package."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert EnterpriseExampleExtender is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert isinstance(EnterpriseExampleExtender, type)


class TestEnterpriseExampleExtenderInheritance:
    """Test that EnterpriseExampleExtender inherits from Extender."""

    def test_inherits_from_extender(self) -> None:
        """Test that EnterpriseExampleExtender is a subclass of Extender."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert issubclass(EnterpriseExampleExtender, Extender)

    def test_instance_is_extender(self) -> None:
        """Test that an instance is an instance of Extender."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        instance = EnterpriseExampleExtender()
        assert isinstance(instance, Extender)


class TestEnterpriseExampleExtenderBasicFunctionality:
    """Test basic functionality of EnterpriseExampleExtender."""

    def test_has_name_attribute(self) -> None:
        """Test that the extender has a name or identifier."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        # Extender implementations should have some form of identification
        instance = EnterpriseExampleExtender()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "EnterpriseExampleExtender"

    def test_can_instantiate(self) -> None:
        """Test that the extender can be instantiated."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        instance = EnterpriseExampleExtender()
        assert instance is not None


class TestEnterpriseExampleExtenderErrorContract:
    """raise_on_error, wraps() and the pass-through __call__."""

    def test_raise_on_error_defaults_to_true(self) -> None:
        """Default is breaking: a failure propagates."""
        assert EnterpriseExampleExtender().raise_on_error is True

    def test_raise_on_error_can_be_disabled(self) -> None:
        """raise_on_error=False marks the extender as warning-only."""
        assert EnterpriseExampleExtender(raise_on_error=False).raise_on_error is False

    def test_wraps_calculate_feature_hook(self) -> None:
        """The extender must hook feature calculation, otherwise it never runs."""
        assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in EnterpriseExampleExtender().wraps()

    def test_call_is_pass_through(self) -> None:
        """__call__ forwards args and returns the wrapped function's result."""
        assert EnterpriseExampleExtender()(lambda x, y: x + y, 1, 2) == 3


class TestEnterpriseExampleExtenderRunAll:
    """End-to-end raise_on_error semantics through mloda.run_all."""

    def test_pass_through_extender_does_not_change_result(self) -> None:
        """The example extender is a no-op: the feature computes normally."""
        assert _run_value_int(EnterpriseExampleExtender()) == _VALUE_INT

    def test_failing_extender_breaks_run_by_default(self) -> None:
        """raise_on_error=True (default): the extender failure propagates out of run_all."""
        with pytest.raises(Exception, match="extender boom"):
            _run_value_int(FailingEnterpriseExampleExtender())

    def test_failing_extender_warns_only_when_raise_on_error_false(self, caplog: pytest.LogCaptureFixture) -> None:
        """raise_on_error=False: failure is logged as a warning and the feature still computes."""
        with caplog.at_level(logging.WARNING):
            values = _run_value_int(FailingEnterpriseExampleExtender(raise_on_error=False))

        assert values == _VALUE_INT

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("extender boom" in message for message in warnings), warnings
