"""Both extender test_manifest.py files must share one manifest contract mixin, not near-duplicate copies."""

from __future__ import annotations

import importlib
import inspect

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin

_MANIFEST_TEST_MODULES = [
    "mloda.community.extenders.openlineage.tests.test_manifest",
    "mloda.community.extenders.otel.tests.test_manifest",
]


def test_manifest_test_modules_define_exactly_one_mixin_subclass_and_no_module_functions() -> None:
    for module_name in _MANIFEST_TEST_MODULES:
        module = importlib.import_module(module_name)

        classes = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module_name and issubclass(obj, OptionalDependencyPackageTestMixin)
        ]
        assert len(classes) == 1, f"{module_name} must define exactly one OptionalDependencyPackageTestMixin subclass"

        functions = [
            name
            for name, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module_name and name.startswith("test_")
        ]
        assert functions == [], f"{module_name} must not define module-level test_ functions: {functions}"
