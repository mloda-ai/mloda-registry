"""Tests for mloda.testing package."""

import ast
import importlib
import importlib.util
import pkgutil
import subprocess  # nosec
import sys
from typing import Any

import mloda._testing
import pytest
from mloda.steward import Extender, ExtenderHook

from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.import_isolation import block_root, evict_package


def test_feature_group_test_base_import() -> None:
    """Verify FeatureGroupTestBase can be imported."""
    assert FeatureGroupTestBase is not None


def test_import_does_not_load_pytest_outside_pytest() -> None:
    """Importing mloda.testing must not import pytest in a process where pytest is not loaded."""
    result = subprocess.run(  # nosec
        [sys.executable, "-c", "import sys, mloda.testing; print('pytest' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


_WITHOUT_PYARROW_PROBE = """
import sys

sys.modules["pyarrow"] = None

from typing import Any

import mloda.testing
import mloda.testing.feature_group_shape
import mloda.testing.framework_adapter
import mloda.testing.operation_group_shape
import mloda.testing.typed_data
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import DataType, Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

from mloda.testing.compute_framework_shape import check_compute_framework_shape
from mloda.testing.feature_group_shape import check_feature_group_shape
from mloda.testing.typed_data import TypedTestData


class DoubledFeature(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"x_doubled"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed("x")}

    @classmethod
    def compute_framework_rule(cls) -> Any:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"x_doubled": [None if v is None else v * 2 for v in data["x"]]}


group = check_feature_group_shape(DoubledFeature, input_data=TypedTestData({"x": (DataType.INT64, [1, 2, 3])}))
framework = check_compute_framework_shape(PythonDictFramework)
print(group.failures, framework.failures, sys.modules["pyarrow"])
"""


def test_kit_works_without_pyarrow() -> None:
    """With pyarrow blocked, the pyarrow-free kit modules import and the PythonDict shape checks pass."""
    result = subprocess.run([sys.executable, "-c", _WITHOUT_PYARROW_PROBE], capture_output=True, text=True)  # nosec
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "() () None"


@pytest.mark.parametrize(
    ("module", "guarded_package"),
    [
        ("mloda.testing.feature_groups.data_operations.base", "mloda.testing.feature_groups.data_operations"),
        ("mloda.testing.extenders.contract", "mloda.testing.extenders"),
    ],
)
def test_pyarrow_guarded_package_skips_without_pyarrow(
    monkeypatch: pytest.MonkeyPatch, module: str, guarded_package: str
) -> None:
    """A module under a pyarrow-guarded package is skipped, not an ImportError, when pyarrow is missing."""
    evict_package(monkeypatch, guarded_package)
    block_root(monkeypatch, "pyarrow")
    with pytest.raises(pytest.skip.Exception, match=r"requires mloda-testing\[pyarrow\]"):
        importlib.import_module(module)


def test_contract_mixin_assert_reports_compared_values() -> None:
    """Self-test: a failing contract assert should show the compared values, not an empty message."""

    class _OffByOneExtender(Extender):
        def __init__(self) -> None:
            self.raise_on_error = True

        def wraps(self) -> set[ExtenderHook]:
            return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs) + 1

    class _Host(ExtenderContractTestMixin):
        def make_extender(self, *, raise_on_error: bool | None = None) -> _OffByOneExtender:
            return _OffByOneExtender()

    with pytest.raises(AssertionError, match=r"assert 8 == 7"):
        _Host().test_contract_call_returns_wrapped_result_unchanged()


def _core_testing_modules() -> list[str]:
    names = [m.name for m in pkgutil.walk_packages(mloda._testing.__path__, "mloda._testing.")]
    return sorted(name for name in names if not name.rsplit(".", 1)[-1].startswith("_"))


def _public_names_defined(module_name: str) -> set[str]:
    """Core's own __all__ if it declares one, else the public top-level class, function and assignment names."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None, module_name
    with open(spec.origin, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return set(ast.literal_eval(node.value))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


@pytest.mark.parametrize("core_name", _core_testing_modules())
def test_kit_module_reexports_the_public_names_of_its_core_module(core_name: str) -> None:
    """Each core mloda._testing module is mirrored at the same path under mloda.testing, name for name."""
    kit = importlib.import_module(core_name.replace("mloda._testing", "mloda.testing", 1))
    core = importlib.import_module(core_name)
    exported = set(getattr(kit, "__all__", ()))
    assert exported == _public_names_defined(core_name)
    for name in exported:
        assert getattr(kit, name) is getattr(core, name), name


def test_reexported_core_suite_is_assert_rewritten() -> None:
    """Self-test: a core suite reached through the kit gets pytest's assert rewriting, so failures show values."""
    importlib.import_module("mloda.testing.source.read_file_suite")
    core = sys.modules["mloda._testing.source.read_file_suite"]
    assert type(core.__loader__).__name__ == "AssertionRewritingHook"
