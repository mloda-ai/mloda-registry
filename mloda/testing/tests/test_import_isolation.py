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
