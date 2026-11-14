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


# Core modules not mirrored yet.
_NOT_REEXPORTED = frozenset(
    {"mloda._testing.value_fit", "mloda._testing.source.reclaim", "mloda._testing.compute_framework"}
)


def _is_not_reexported(name: str) -> bool:
    return any(name == excluded or name.startswith(excluded + ".") for excluded in _NOT_REEXPORTED)


def _core_testing_modules() -> list[str]:
    names = [m.name for m in pkgutil.walk_packages(mloda._testing.__path__, "mloda._testing.")]
    return sorted(
        name for name in names if not _is_not_reexported(name) and not name.rsplit(".", 1)[-1].startswith("_")
    )


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
