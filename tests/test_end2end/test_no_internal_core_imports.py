"""Repository guard: flags any ``mloda.core`` import under ``mloda/`` (``from mloda.core...
import X`` or a bare ``import mloda.core...``) unless the exact ``(module, name)`` pair is in
the allowlist below. This is a fail-closed check: it flags every non-allowlisted name whether or
not a public equivalent exists, not only names that happen to have one. A bare module import has
no single symbol name to check against the allowlist, so it is always flagged. ``tests/`` is out
of scope since it may deliberately reach into core internals (e.g. plugin/entry-point loading
machinery).
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_ROOT = _REPO_ROOT / "mloda"

_SKIP_DIRS = {"__pycache__", "site-packages", "node_modules", "build", "dist", ".venv"}

# (module, name) pairs with no public mloda.user/mloda.provider equivalent yet. Keyed by the
# exact module too, so a name that merely matches one of these (imported from some other
# mloda.core module) is not silently exempt.
_ALLOWED_INTERNAL_IMPORTS = {
    ("mloda.core.abstract_plugins.components.utils", "contained_raise_log_level"),
    ("mloda.core.abstract_plugins.components.utils", "contained_raise_reason"),
    ("mloda.core.abstract_plugins.components.utils", "escalate_match_abort"),
    ("mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser", "option_key_is_present"),
    ("mloda.core.abstract_plugins.function_extender", "_CompositeExtender"),
    ("mloda.core.abstract_plugins.hook_context", "instrument"),
}


def _is_core_module(name: str) -> bool:
    return name == "mloda.core" or name.startswith("mloda.core.")


def find_internal_core_imports(root: Path) -> list[str]:
    """Return "relpath:lineno: ..." for every non-allowlisted mloda.core import under root."""
    hits: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part.startswith(".") or part in _SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.name == "test_no_internal_core_imports.py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None or not _is_core_module(node.module):
                    continue
                for alias in node.names:
                    if (node.module, alias.name) in _ALLOWED_INTERNAL_IMPORTS:
                        continue
                    hits.append(f"{rel.as_posix()}:{node.lineno}: {alias.name} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_core_module(alias.name):
                        continue
                    # A bare module import has no single symbol name to check against the
                    # allowlist, so any bare `import mloda.core...` is flagged outright.
                    target = f" as {alias.asname}" if alias.asname else ""
                    hits.append(f"{rel.as_posix()}:{node.lineno}: import {alias.name}{target}")
    return sorted(hits)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_migratable_symbol_import_flagged(tmp_path: Path) -> None:
    """A symbol with a public equivalent imported from mloda.core.* is flagged."""
    _write(tmp_path / "m.py", "from mloda.core.abstract_plugins.components.options import Options\n")
    hits = find_internal_core_imports(tmp_path)
    assert len(hits) == 1
    assert "Options" in hits[0]
    assert "m.py" in hits[0]


def test_nested_function_local_import_flagged(tmp_path: Path) -> None:
    """A function-local (nested) mloda.core.* import is also caught, not just module-level ones."""
    body = "def f():\n    from mloda.core.abstract_plugins.components.feature_set import FeatureSet\n    return FeatureSet\n"
    _write(tmp_path / "m.py", body)
    hits = find_internal_core_imports(tmp_path)
    assert len(hits) == 1
    assert "FeatureSet" in hits[0]


def test_allowlisted_internal_name_not_flagged(tmp_path: Path) -> None:
    """One of the still-internal-only allowlisted (module, name) pairs is not flagged."""
    _write(tmp_path / "m.py", "from mloda.core.abstract_plugins.components.utils import contained_raise_log_level\n")
    assert find_internal_core_imports(tmp_path) == []


def test_allowlist_is_keyed_by_module_and_name(tmp_path: Path) -> None:
    """A name that matches an allowlisted symbol but comes from a different module is still flagged."""
    _write(tmp_path / "m.py", "from mloda.core.wrong_module import escalate_match_abort\n")
    hits = find_internal_core_imports(tmp_path)
    assert len(hits) == 1
    assert "escalate_match_abort" in hits[0]
    assert "mloda.core.wrong_module" in hits[0]


def test_bare_module_import_flagged(tmp_path: Path) -> None:
    """A bare `import mloda.core.x` has no allowlisted case and is always flagged."""
    _write(tmp_path / "m.py", "import mloda.core.abstract_plugins.components.options\n")
    hits = find_internal_core_imports(tmp_path)
    assert len(hits) == 1
    assert "mloda.core.abstract_plugins.components.options" in hits[0]


def test_public_user_import_not_flagged(tmp_path: Path) -> None:
    """A public mloda.user import is not flagged."""
    _write(tmp_path / "m.py", "from mloda.user import Options\n")
    assert find_internal_core_imports(tmp_path) == []


def test_public_provider_import_not_flagged(tmp_path: Path) -> None:
    """A public mloda.provider import is not flagged."""
    _write(tmp_path / "m.py", "from mloda.provider import FeatureSet\n")
    assert find_internal_core_imports(tmp_path) == []


def test_guard_module_self_excluded(tmp_path: Path) -> None:
    """A file named like this guard module is skipped even with a flaggable import."""
    _write(
        tmp_path / "test_no_internal_core_imports.py",
        "from mloda.core.abstract_plugins.components.options import Options\n",
    )
    assert find_internal_core_imports(tmp_path) == []


def test_build_artifact_dirs_not_scanned(tmp_path: Path) -> None:
    """A flaggable file under a build artifact directory is not scanned."""
    body = "from mloda.core.abstract_plugins.components.options import Options\n"
    _write(tmp_path / "build" / "lib" / "m.py", body)
    _write(tmp_path / "dist" / "m.py", body)
    _write(tmp_path / "pkg.egg-info" / "m.py", body)
    _write(tmp_path / ".venv" / "lib" / "m.py", body)
    assert find_internal_core_imports(tmp_path) == []


def test_mloda_package_is_clean() -> None:
    """The mloda/ package must currently have zero non-allowlisted mloda.core imports."""
    offenders = find_internal_core_imports(_SCAN_ROOT)
    assert offenders == [], "Non-allowlisted mloda.core import found:\n" + "\n".join(offenders)
