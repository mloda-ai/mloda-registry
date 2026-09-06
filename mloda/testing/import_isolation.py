"""Test-only helpers for poisoning and evicting sys.modules entries with pytest.MonkeyPatch."""

from __future__ import annotations

import sys

import pytest


def block_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Poison sys.modules so any import of ``root`` (or a cached submodule) raises ModuleNotFoundError."""
    names = {root} | {name for name in sys.modules if name == root or name.startswith(f"{root}.")}
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def evict_package(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Cold-evict ``dotted`` (and its manifest) from sys.modules and detach it from its parent package."""
    parent_name, _, leaf = dotted.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        # setattr then delattr queues two undo entries against the same (parent, leaf) pair, so
        # teardown overwrites whatever a cold import inside the test re-binds on the parent.
        monkeypatch.setattr(parent, leaf, None, raising=False)
        monkeypatch.delattr(parent, leaf, raising=False)

    for name in list(sys.modules):
        if name == dotted or name.startswith(f"{dotted}."):
            monkeypatch.setitem(sys.modules, name, sys.modules[name])
            monkeypatch.delitem(sys.modules, name)

    for name in (dotted, f"{dotted}.manifest"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name, raising=False)
