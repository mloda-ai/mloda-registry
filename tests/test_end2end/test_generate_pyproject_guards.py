"""Robustness guards for scripts/generate_pyproject.py.

The generator is the single source of truth for every package's
``pyproject.toml`` and for the root mloda-core pin. Three silent-failure modes
must be turned into loud failures:

Guard 1 -- a missing ``[defaults].core_dependency`` must raise, not silently
substitute ``""`` and emit ``dependencies = [""]``.

Guard 2 -- write mode must exit non-zero when the root mloda-core dependency
entry cannot be synced, instead of returning 0 and leaving a stale pin.

Guard 3 -- a meta-package (``workspace_deps``) flagged ``py_typed`` must raise,
instead of emitting ``packages = []`` and a wheel without its PEP 561 marker.

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an
installed package), so it is loaded here by file path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

gen = load_script("generate_pyproject", _GEN_PATH)


def test_generate_raises_when_core_dependency_missing() -> None:
    """generate_pyproject must fail loudly if core_dependency is absent.

    Today ``defaults.get("core_dependency", "")`` turns a missing key into an
    empty string, so ``"{core_dependency}"`` placeholders collapse to ``""``
    and packages get an invalid ``dependencies = [""]``. This must raise
    ``ValueError`` instead.
    """
    shared, packages_config = gen.load_configs()
    packages = packages_config.get("packages", {})
    assert "core_dependency" in shared["defaults"], "fixture assumption: shared config defines core_dependency"

    # Simulate a misconfigured shared.toml with the pin removed.
    shared["defaults"].pop("core_dependency", None)

    pkg_config = packages["mloda-registry"]
    assert "{core_dependency}" in pkg_config.get("dependencies", []), (
        "fixture assumption: mloda-registry deps use the {core_dependency} placeholder"
    )

    with pytest.raises(ValueError):
        content = gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)
        # Belt and suspenders: if it did NOT raise, it must at least not have
        # emitted an empty dependency entry. This makes the current failure
        # mode (silent "") produce a descriptive assertion rather than a bare
        # "DID NOT RAISE".
        assert '[""]' not in content and 'dependencies = [""]' not in content, (
            f"generate_pyproject silently produced an empty core dependency:\n{content}"
        )


def test_write_mode_exits_nonzero_when_root_entry_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write mode must return non-zero when the root mloda-core entry is unsyncable.

    The generator's file-path globals are CWD-relative, so an isolated
    workspace under ``tmp_path`` (real configs copied in, a root
    ``pyproject.toml`` with no ``mloda`` core entry) fully sandboxes the run.
    ``update_root_core_dependency`` then returns ``(False, ...)``. Today
    ``main()`` ignores that in write mode and returns 0; it must return 1.
    """
    # Snapshot the real root pyproject to prove the run never touches the repo.
    real_root_before = _ROOT_PYPROJECT.read_text()

    # Build an isolated workspace: real configs, sandboxed root pyproject.
    (tmp_path / "config").mkdir()
    shutil.copy(_REPO_ROOT / "config" / "shared.toml", tmp_path / "config" / "shared.toml")
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", tmp_path / "config" / "packages.toml")

    # Root pyproject deliberately lacks any ``mloda`` core dependency entry,
    # so update_root_core_dependency cannot find one to sync.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox-root"\ndependencies = ["requests>=2.0"]\n',
    )

    # CWD-relative globals now resolve inside the sandbox.
    monkeypatch.chdir(tmp_path)
    # Write mode: no --check flag.
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py"])

    return_code = gen.main()

    assert return_code == 1, (
        "generate_pyproject.main() in write mode must exit non-zero when the root "
        f"mloda-core dependency entry cannot be synced, but it returned {return_code!r}."
    )

    # The real repository must be untouched by the sandboxed run.
    assert _ROOT_PYPROJECT.read_text() == real_root_before, (
        "the sandboxed generator run modified the real repository root pyproject.toml"
    )


def _meta_package_config(py_typed: bool | None = None) -> dict[str, Any]:
    """Synthetic meta-package config, optionally flagged ``py_typed``."""
    pkg_config: dict[str, Any] = {
        "description": "Meta package aggregating workspace members",
        "path": "mloda/meta",
        "dependencies": [],
        "workspace_deps": ["mloda-registry"],
    }
    if py_typed is not None:
        pkg_config["py_typed"] = py_typed
    return pkg_config


def test_generate_raises_when_meta_package_is_flagged_py_typed() -> None:
    """The ``workspace_deps`` branch emits ``packages = []`` and never consults ``py_typed``."""
    shared, _packages_config = gen.load_configs()
    pkg_config = _meta_package_config(py_typed=True)
    all_packages: dict[str, dict[str, Any]] = {"mloda-meta": pkg_config}

    with pytest.raises(ValueError, match="py_typed"):
        gen.generate_pyproject("mloda-meta", pkg_config, shared, all_packages)


def test_meta_package_without_py_typed_still_generates() -> None:
    """The guard fires on the combination only, not on every meta-package."""
    shared, _packages_config = gen.load_configs()
    pkg_config = _meta_package_config()
    all_packages: dict[str, dict[str, Any]] = {"mloda-meta": pkg_config}

    content: str = gen.generate_pyproject("mloda-meta", pkg_config, shared, all_packages)

    assert "packages = []" in content, content
    assert "package-data" not in content, content
