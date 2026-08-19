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
behaviour directly, including that an import inside a module-level ``except:``
handler is walked, and the real-tree test also asserts the walk is non-vacuous
so a moved or emptied ``data_operations`` tree cannot make the guard pass
silently. The guard only sees direct top-level third-party imports under
``data_operations/**``; an optional framework reached only indirectly through
mloda core's own compute-framework shims (e.g. duckdb, which has no direct
import anywhere under ``data_operations/**`` and is only ever imported inside
``mloda_plugins.compute_framework.base_implementations.duckdb.*``) is invisible
to this guard and relies on core's own try/except shims guarding it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations import manifest_utils
from mloda.community.feature_groups.data_operations.manifest_utils import load_plugin_classes

_IMPORT_MODULE_TARGET = "mloda.community.feature_groups.data_operations.manifest_utils.importlib.import_module"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_OPERATIONS_ROOT = _REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"

# mloda and mloda_plugins are required core dependencies, not optional backends.
_FIRST_PARTY_IMPORT_ROOTS = frozenset({"mloda", "mloda_plugins"})


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


def _trusted_type_checking_alias(tree: ast.Module) -> str | None:
    """The module-level name that genuinely aliases ``typing.TYPE_CHECKING``, or ``None``.

    Only a bare ``from typing import TYPE_CHECKING [as alias]`` at module level earns trust, and
    only if nothing else at module level rebinds that name. Without this, a module could spoof the
    guard's exemption with ``TYPE_CHECKING = True`` and hide a real, always-executed import from it.
    """
    alias_name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    alias_name = alias.asname or alias.name
    if alias_name is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "typing":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == alias_name and isinstance(child.ctx, ast.Store):
                return None
    return alias_name


def _module_level_import_roots(tree: ast.Module) -> set[str]:
    """Import roots from module-scope import statements only, skipping def/class bodies."""
    trusted_alias = _trusted_type_checking_alias(tree)
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
        elif isinstance(node, ast.If):
            is_type_checking = (
                isinstance(node.test, ast.Name) and trusted_alias is not None and node.test.id == trusted_alias
            ) or (
                isinstance(node.test, ast.Attribute)
                and node.test.attr == "TYPE_CHECKING"
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "typing"
            )
            if is_type_checking:
                stack.extend(node.orelse)
                continue
            stack.extend(ast.iter_child_nodes(node))
        else:
            stack.extend(ast.iter_child_nodes(node))
    return roots


def third_party_import_roots_by_file(root_dir: Path) -> dict[str, list[Path]]:
    """Map each non-stdlib, non-mloda top-level import root under root_dir to the files that use it."""
    stdlib_roots = set(sys.stdlib_module_names)
    # Bridge stdlib drift across the 3.10-3.14 support window (e.g. PEP 594 removals, tomllib addition).
    stdlib_roots.update(
        {
            "aifc",
            "audioop",
            "cgi",
            "cgitb",
            "chunk",
            "crypt",
            "imghdr",
            "mailcap",
            "msilib",
            "nis",
            "nntplib",
            "ossaudiodev",
            "pipes",
            "smtpd",
            "sndhdr",
            "spwd",
            "sunau",
            "telnetlib",
            "tomllib",
            "uu",
            "xdrlib",
        }
    )
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
        pytest.param(
            "TYPE_CHECKING = True\nif TYPE_CHECKING:\n    import numpy\n",
            "numpy",
            id="spoofed_type_checking_name_is_not_exempt",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\nif TYPE_CHECKING:\n    import numpy\n",
            "numpy",
            id="reassigned_type_checking_name_is_not_exempt",
        ),
        pytest.param(
            "class C:\n    TYPE_CHECKING = True\n\n\nif C.TYPE_CHECKING:\n    import numpy\n",
            "numpy",
            id="spoofed_type_checking_attr_is_not_exempt",
        ),
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
        pytest.param(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\n",
            id="import_in_type_checking_name",
        ),
        pytest.param(
            "import typing\nif typing.TYPE_CHECKING:\n    import numpy\n",
            id="import_in_type_checking_attr",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING as TC\nif TC:\n    import numpy\n",
            id="import_in_type_checking_aliased_name",
        ),
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
    an ``ast.stmt``. This pins the fixed walker's behaviour: it descends into the handler
    body and catches the fallback root.
    """
    body = "try:\n    import polars as pl\nexcept ImportError:\n    import pandas as pl\n"
    _write(tmp_path / "m.py", body)
    files_by_root = third_party_import_roots_by_file(tmp_path)
    assert files_by_root == {
        "polars": [Path("m.py")],
        "pandas": [Path("m.py")],
    }


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
