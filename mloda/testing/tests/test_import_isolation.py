"""Direct unit test for mloda.testing.import_isolation.evict_package."""

from __future__ import annotations

import importlib
import sys
import uuid
from pathlib import Path

import pytest

from mloda.testing.import_isolation import evict_package


def _write_throwaway_package(base: Path, name: str) -> None:
    package_dir = base / name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "leaf.py").write_text("MARKER = 'loaded'\n")


def test_cold_imported_leaf_and_parent_attribute_are_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    monkeypatch.syspath_prepend(str(tmp_path))

    parent = importlib.import_module(package_name)
    assert not hasattr(parent, "leaf")

    leaf_name = f"{package_name}.leaf"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, leaf_name)
        leaf = importlib.import_module(leaf_name)

        assert leaf.MARKER == "loaded"
        assert leaf_name in sys.modules
        assert getattr(parent, "leaf", None) is leaf

    assert leaf_name not in sys.modules
    assert not hasattr(parent, "leaf")


def test_cold_imported_sibling_submodule_is_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling of the evicted target (not `dotted` itself, not its `.manifest`) cold-imported
    inside the same context must not leak into sys.modules once the context exits."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    (tmp_path / package_name / "extra.py").write_text("MARKER = 'extra'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    leaf_name = f"{package_name}.leaf"
    extra_name = f"{package_name}.extra"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, leaf_name)
        importlib.import_module(leaf_name)
        extra = importlib.import_module(extra_name)

        assert extra.MARKER == "extra"
        assert extra_name in sys.modules

    assert extra_name not in sys.modules


def test_cold_imported_submodule_under_the_evicted_package_is_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `dotted` is itself a package, a submodule under it cold-imported inside the same
    context must not leak into sys.modules once the context exits."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    sub_dir = package_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inner.py").write_text("MARKER = 'inner'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    sub_name = f"{package_name}.sub"
    inner_name = f"{sub_name}.inner"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, sub_name)
        inner = importlib.import_module(inner_name)

        assert inner.MARKER == "inner"
        assert inner_name in sys.modules

    assert inner_name not in sys.modules
