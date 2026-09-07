"""Test-only helpers for poisoning and evicting sys.modules entries with pytest.MonkeyPatch."""

from __future__ import annotations

import importlib.util
import pkgutil
import sys

import pytest


def block_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Poison sys.modules so any import of ``root`` (or a cached submodule) raises ModuleNotFoundError."""
    names = {root} | {name for name in sys.modules if name == root or name.startswith(f"{root}.")}
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def _sibling_submodule_names(parent_name: str) -> set[str]:
    """Every submodule name findable under ``parent_name`` on disk, without importing it.

    Only names not yet in sys.modules are returned: already-loaded siblings are left alone,
    while names that could still be cold-imported during the test are pre-registered for cleanup.
    """
    if not parent_name:
        return set()

    spec = importlib.util.find_spec(parent_name)
    if spec is None or spec.submodule_search_locations is None:
        return set()

    names = set()
    for module_info in pkgutil.iter_modules(spec.submodule_search_locations):
        full_name = f"{parent_name}.{module_info.name}"
        if full_name not in sys.modules:
            names.add(full_name)
    return names


def evict_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Cold-evict every sys.modules entry at or under ``root``, forcing a genuine cold re-import (unlike
    ``block_root``, which poisons entries to raise ModuleNotFoundError instead).

    The setitem-then-delitem dance queues two undo entries per name, so teardown restores the pre-test
    module object even if the test's cold re-import rebinds the name to a new one.
    """
    for name in list(sys.modules):
        if name == root or name.startswith(f"{root}."):
            monkeypatch.setitem(sys.modules, name, sys.modules[name])
            monkeypatch.delitem(sys.modules, name)


def evict_package(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Cold-evict ``dotted`` (and its manifest) from sys.modules and detach it from its parent package.

    Also pre-registers removal for every sibling submodule the parent package could still cold-import,
    and every submodule ``dotted`` itself could cold-import, during the test, so none of them leak.
    """
    parent_name, _, leaf = dotted.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        # setattr then delattr queues two undo entries against the same (parent, leaf) pair, so
        # teardown overwrites whatever a cold import inside the test re-binds on the parent.
        monkeypatch.setattr(parent, leaf, None, raising=False)
        monkeypatch.delattr(parent, leaf, raising=False)

    evict_root(monkeypatch, dotted)

    candidates = (
        {dotted, f"{dotted}.manifest"} | _sibling_submodule_names(parent_name) | _sibling_submodule_names(dotted)
    )
    for name in candidates:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name, raising=False)
