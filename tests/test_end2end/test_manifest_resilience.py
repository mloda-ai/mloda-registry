"""Unit tests for the resilient manifest loader helper (issue #271).

``load_plugin_classes`` builds an entry-point manifest's class list by importing
each backend module individually. A backend whose optional compute framework
(pandas, polars, duckdb, pyarrow, numpy) is not installed must be skipped so the
rest still register, while any other import error must stay loud. numpy is
transitive: it is only ever missing because pandas needs it, not because a
backend module targets a "numpy compute framework" directly. These tests drive
that behaviour by monkeypatching the helper module's ``importlib.import_module``,
so they need no optional framework installed and touch no network.

A further test walks the real data_operations tree on disk and asserts every
module's top-level third-party import root is covered by ``_OPTIONAL_BACKENDS``,
so a backend module that top-imports an uncovered root fails this test instead
of silently breaking a bare floor install. ``third_party_import_roots_by_file``
is the public scanner backing that guard; a ``tmp_path`` battery below pins its
behaviour directly. The guard only sees direct top-level third-party imports under
``data_operations/**``; an optional framework reached only indirectly through
mloda core's own compute-framework shims (e.g. duckdb, which has no direct
import anywhere under ``data_operations/**`` and is only ever imported inside
``mloda_plugins.compute_framework.base_implementations.duckdb.*``) is invisible
to this guard and relies on core's own try/except shims guarding it. A second guard
cross-validates the allowlist against ``config/packages.toml``'s declared optional
frameworks, so a required dependency cannot be silently marked optional.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from mloda.community.feature_groups.data_operations import manifest_utils
from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes

_IMPORT_MODULE_TARGET = "mloda.community.feature_groups.data_operations.manifest_utils.importlib.import_module"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_OPERATIONS_ROOT = _REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"
_DATA_OPERATIONS_REL = _DATA_OPERATIONS_ROOT.relative_to(_REPO_ROOT).as_posix()
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# mloda and mloda_plugins are required core dependencies, not optional backends.
_FIRST_PARTY_IMPORT_ROOTS = frozenset({"mloda", "mloda_plugins"})

# numpy is only ever missing because pandas needs it, never its own optional extra.
_TRANSITIVE_ONLY_OPTIONAL_ROOTS = frozenset({"numpy"})

# Extras that never name a compute framework: 'all' just re-aggregates the others; 'dev' is
# tooling (see config/shared.toml [defaults].optional_dependencies and mloda-testing's own
# 'dev' extra in this file), never a compute-framework backend.
_NON_FRAMEWORK_EXTRAS = frozenset({"all", "dev"})


class _KeptClass:
    """Placeholder class returned by a successfully imported backend."""


def test_skips_backend_with_missing_optional_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    kept_module = SimpleNamespace(KeptClass=_KeptClass)

    def fake_import(name: str) -> Any:
        if name.endswith("polars_backend"):
            raise ModuleNotFoundError("No module named 'polars'", name="polars")
        return kept_module

    monkeypatch.setattr(_IMPORT_MODULE_TARGET, fake_import)

    classes = load_plugin_classes(
        "pkg",
        [
            ("polars_backend", "PolarsClass"),
            ("pandas_backend", "KeptClass"),
        ],
    )

    assert [c.__name__ for c in classes] == ["_KeptClass"]


def test_skips_backend_with_missing_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    kept_module = SimpleNamespace(KeptClass=_KeptClass)

    def fake_import(name: str) -> Any:
        if name.endswith("pandas_binning"):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
        return kept_module

    monkeypatch.setattr(_IMPORT_MODULE_TARGET, fake_import)

    classes = load_plugin_classes(
        "pkg",
        [
            ("pandas_binning", "PandasBinningClass"),
            ("pandas_backend", "KeptClass"),
        ],
    )

    assert [c.__name__ for c in classes] == ["_KeptClass"]


def test_reraises_non_optional_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> ModuleType:
        raise ModuleNotFoundError(
            "No module named 'mloda.community.foo.bar'",
            name="mloda.community.foo.bar",
        )

    monkeypatch.setattr(_IMPORT_MODULE_TARGET, fake_import)

    with pytest.raises(ModuleNotFoundError):
        load_plugin_classes("pkg", [("missing_backend", "MissingClass")])


def test_empty_specs_returns_empty_list() -> None:
    assert load_plugin_classes("pkg", []) == []


def _module_level_import_roots(tree: ast.Module) -> set[str]:
    """Import roots from module-scope import statements only, skipping def/class bodies."""
    roots: set[str] = set()
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            stack.extend(ast.iter_child_nodes(node))
    return roots


def third_party_import_roots_by_file(root_dir: Path) -> dict[str, list[Path]]:
    """Map each non-stdlib, non-mloda top-level import root under root_dir to the files that use it."""
    stdlib_roots = set(sys.stdlib_module_names)
    files_by_root: dict[str, list[Path]] = {}
    for py_file in sorted(root_dir.rglob("*.py")):
        rel_path = py_file.relative_to(root_dir)
        if "tests" in rel_path.parts or "__pycache__" in rel_path.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for root in _module_level_import_roots(tree):
            if root in stdlib_roots or root in _FIRST_PARTY_IMPORT_ROOTS:
                continue
            files_by_root.setdefault(root, []).append(rel_path)
    return files_by_root


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


@pytest.mark.parametrize(
    "body,expected_root",
    [
        pytest.param("import numpy\n", "numpy", id="plain_import"),
        pytest.param("from pandas import DataFrame\n", "pandas", id="import_from"),
        pytest.param("from polars import *\n", "polars", id="star_import"),
    ],
)
def test_third_party_import_roots_by_file_catches_module_level_import(
    body: str, expected_root: str, tmp_path: Path
) -> None:
    _write(tmp_path / "m.py", body)
    files_by_root = third_party_import_roots_by_file(tmp_path)
    assert files_by_root == {expected_root: [Path("m.py")]}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("def f() -> None:\n    import numpy\n", id="import_in_function_body"),
        pytest.param("class C:\n    import numpy\n", id="import_in_class_body"),
        pytest.param("from . import x\n", id="relative_import_dot"),
        pytest.param("from ..pkg import x\n", id="relative_import_dotdot"),
        pytest.param("import json\n", id="stdlib_import"),
        pytest.param("import mloda_plugins.something\n", id="first_party_import"),
    ],
)
def test_third_party_import_roots_by_file_ignores_out_of_scope_import(body: str, tmp_path: Path) -> None:
    _write(tmp_path / "m.py", body)
    assert third_party_import_roots_by_file(tmp_path) == {}


def test_third_party_import_roots_by_file_catches_both_roots_like_pandas_binning(tmp_path: Path) -> None:
    """Regression shape: numpy imported before pandas, both top-level in one file (see pandas_binning.py)."""
    body = "import numpy as np\nimport pandas as pd\n"
    _write(tmp_path / "pandas_binning.py", body)
    files_by_root = third_party_import_roots_by_file(tmp_path)
    assert files_by_root == {
        "numpy": [Path("pandas_binning.py")],
        "pandas": [Path("pandas_binning.py")],
    }


def test_except_handler_fallback_import_root_is_caught(tmp_path: Path) -> None:
    """An import inside a module-level ``except ImportError:`` handler must be caught.

    mloda core's framework shims use ``try: import <framework> / except ImportError: ...``;
    if the except body itself falls back to importing a *different* optional root, that
    fallback import lives only inside the ``ExceptHandler`` node. A naive walk that only
    recurses into ``ast.stmt`` children would miss it, since ``ast.ExceptHandler`` is not
    an ``ast.stmt``.
    """
    body = "try:\n    import polars as pl\nexcept ImportError:\n    import pandas as pl\n"
    _write(tmp_path / "m.py", body)
    files_by_root = third_party_import_roots_by_file(tmp_path)
    assert files_by_root == {
        "polars": [Path("m.py")],
        "pandas": [Path("m.py")],
    }


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\s\[(@]", spec.strip(), maxsplit=1)[0]


def _declared_optional_framework_roots() -> set[str]:
    """PyPI distribution names the data_operations packages declare as optional extras in packages.toml.

    Compared directly against manifest_utils._OPTIONAL_BACKENDS' import roots; this only works because
    today's optional frameworks share their distribution name with their import name. A future optional
    framework whose distribution name differs from its import root (e.g. scikit-learn -> sklearn) needs
    its own explicit mapping, not a _TRANSITIVE_ONLY_OPTIONAL_ROOTS entry (reserved for genuinely
    transitive dependencies, e.g. numpy via pandas).
    """
    assert _PACKAGES_CONFIG.is_file(), f"packages config not found at {_PACKAGES_CONFIG}"
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG).get("packages", {})
    roots: set[str] = set()
    for pkg_config in packages.values():
        path = pkg_config.get("path", "")
        if path != _DATA_OPERATIONS_REL and not path.startswith(_DATA_OPERATIONS_REL + "/"):
            continue
        for extra_name, specs in pkg_config.get("optional_dependencies", {}).items():
            if extra_name in _NON_FRAMEWORK_EXTRAS:
                continue
            assert isinstance(specs, list), (
                f"{pkg_config.get('path')}: optional_dependencies.{extra_name} must be a list, got {specs!r}"
            )
            roots.update(_dep_name(spec) for spec in specs if "{" not in spec)
    return roots


def test_data_operations_import_roots_are_covered_by_optional_backends() -> None:
    assert _DATA_OPERATIONS_ROOT.is_dir(), f"data_operations tree not found at {_DATA_OPERATIONS_ROOT}"

    files_by_root = third_party_import_roots_by_file(_DATA_OPERATIONS_ROOT)
    assert files_by_root, "walk found no third-party import roots under data_operations; guard is vacuous"

    missing = sorted(set(files_by_root) - manifest_utils._OPTIONAL_BACKENDS)
    assert not missing, (
        "manifest_utils._OPTIONAL_BACKENDS is missing root(s) top-imported under data_operations: "
        + "; ".join(f"{root} (used by {', '.join(str(p) for p in files_by_root[root])})" for root in missing)
        + "; add it to _OPTIONAL_BACKENDS only if it is an optional framework or a transitive dependency of one, "
        "a required dependency must stay a hard failure."
    )


def test_optional_backends_matches_packages_config_declared_frameworks() -> None:
    """_OPTIONAL_BACKENDS must equal packages.toml's declared optional frameworks plus documented transitive roots.

    Catches a required dependency being silently added to the allowlist: any root that is neither
    declared optional in config/packages.toml nor in _TRANSITIVE_ONLY_OPTIONAL_ROOTS fails this test.
    """
    expected = _declared_optional_framework_roots() | _TRANSITIVE_ONLY_OPTIONAL_ROOTS
    unexpected = manifest_utils._OPTIONAL_BACKENDS - expected
    missing = expected - manifest_utils._OPTIONAL_BACKENDS
    assert not unexpected and not missing, (
        f"manifest_utils._OPTIONAL_BACKENDS diverges from config/packages.toml's declared optional "
        f"frameworks plus transitive roots. "
        f"In _OPTIONAL_BACKENDS but not declared optional or transitive (may be a required dependency "
        f"added to the allowlist by mistake): {sorted(unexpected)}. "
        f"Declared optional in config/packages.toml but missing from _OPTIONAL_BACKENDS (add it there so "
        f"its backend is skipped when not installed): {sorted(missing)}."
    )
