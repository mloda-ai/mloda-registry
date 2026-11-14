"""Tests for mloda.testing package."""

import ast
import importlib
import importlib.util
import os
import pkgutil
import re
import subprocess  # nosec
import sys
from pathlib import Path
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

import importlib
from typing import Any

import mloda.testing

for kit_module in sys.argv[1:]:
    importlib.import_module(kit_module)
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


group = check_feature_group_shape(
    DoubledFeature, input_data=TypedTestData({"x": (DataType.INT64, [1, 2, 3])}), require_run=True
)
framework = check_compute_framework_shape(PythonDictFramework)
assert not [name for name in sys.modules if name.startswith("pyarrow.")]
print(group.failures, framework.failures, sys.modules["pyarrow"])
"""


def test_kit_works_without_pyarrow() -> None:
    """With pyarrow blocked, every mirror of a core testing module imports and the PythonDict shape checks pass."""
    mirrors = [name.replace("mloda._testing", "mloda.testing", 1) for name in _core_testing_modules()]
    command = [sys.executable, "-c", _WITHOUT_PYARROW_PROBE, *mirrors]
    result = subprocess.run(command, capture_output=True, text=True)  # nosec
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


_OUTSIDE_PYTEST_PROBE = """
import importlib
import pkgutil
import sys

sys.modules["pyarrow"] = None

import mloda.testing

GUARDED = ("mloda.testing.feature_groups.data_operations", "mloda.testing.extenders")
for package in GUARDED:
    try:
        importlib.import_module(package)
    except ModuleNotFoundError as exc:
        assert exc.name == "pyarrow", exc.name
        assert "requires mloda-testing[pyarrow]" in str(exc), str(exc)
    else:
        raise AssertionError(package + " imported without pyarrow")
    for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
        del sys.modules[name]

failed = []
list(pkgutil.walk_packages(mloda.testing.__path__, "mloda.testing.", onerror=failed.append))
assert set(GUARDED) <= set(failed), failed
print("ok")
"""


def test_pyarrow_guard_raises_module_not_found_outside_pytest() -> None:
    """Outside pytest the guard raises ModuleNotFoundError."""
    result = subprocess.run([sys.executable, "-c", _OUTSIDE_PYTEST_PROBE], capture_output=True, text=True)  # nosec
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


_PYTEST_MAIN_PROBE = (
    "import sys; sys.modules['pyarrow'] = None; import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
)

_GUARDED_IMPORT = "import mloda.testing.extenders.contract\n"

_ROOT_CONFTEST_LAYOUT = {
    "conftest.py": _GUARDED_IMPORT,
    "test_trivial.py": "def test_trivial() -> None:\n    pass\n",
}

_SKIPS_LAYOUT = {
    "conftest.py": (
        "import pytest\n\n\n@pytest.fixture\ndef guarded() -> None:\n"
        "    import mloda.testing.feature_groups.data_operations.base\n"
    ),
    "test_fixture.py": "def test_fixture_skip(guarded: None) -> None:\n    pass\n",
    "test_top.py": _GUARDED_IMPORT + "\n\ndef test_top() -> None:\n    pass\n",
    "sub/conftest.py": "import mloda.testing.feature_groups.data_operations.base\n",
    "sub/test_sub.py": "def test_sub() -> None:\n    pass\n",
}


def _run_pytest_without_pyarrow(project: Path, files: dict[str, str]) -> subprocess.CompletedProcess[str]:
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    ini = project / "empty.ini"
    ini.write_text("[pytest]\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    command = [
        sys.executable,
        "-c",
        _PYTEST_MAIN_PROBE,
        "-p",
        "no:cacheprovider",
        "-p",
        "no:xdist",
        "-c",
        str(ini),
        "--rootdir",
        str(project),
        "-rs",
        str(project),
    ]
    return subprocess.run(command, capture_output=True, text=True, env=env)  # nosec


def test_pyarrow_guard_in_root_conftest_is_a_usage_error(tmp_path: Path) -> None:
    """A root conftest importing a guarded module without pyarrow is a usage error."""
    result = _run_pytest_without_pyarrow(tmp_path, _ROOT_CONFTEST_LAYOUT)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "requires mloda-testing[pyarrow]" in result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr


def test_pyarrow_guard_skips_during_collection_and_tests(tmp_path: Path) -> None:
    """Nested conftests, test modules and fixtures skip when pyarrow is missing."""
    result = _run_pytest_without_pyarrow(tmp_path, _SKIPS_LAYOUT)
    output = result.stdout
    assert result.returncode == 0, output + result.stderr
    skip_lines = [line for line in output.splitlines() if line.startswith("SKIPPED")]
    assert all("requires mloda-testing[pyarrow]" in line for line in skip_lines), skip_lines
    assert any("test_fixture.py" in line for line in skip_lines), skip_lines
    counts = [int(m.group(1)) for line in skip_lines if (m := re.match(r"SKIPPED \[(\d+)\]", line))]
    assert sum(counts) == 3, skip_lines  # fixture test, top-level module, sub directory


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
