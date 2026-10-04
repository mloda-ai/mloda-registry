"""Shared test helpers for the run manifest tests (not collected)."""

from __future__ import annotations

import builtins
from types import ModuleType
from typing import Any

import pytest

from mloda.enterprise.extenders.audit import _quarantine, _seal_index, _segments, _signers, _verify, run_manifest

_SOURCE_MODULES: tuple[ModuleType, ...] = (_signers, _verify, _seal_index, _segments, _quarantine, run_manifest)


def _patch_bindings(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """Patch `name` on every source module whose global is the same object as the original.

    A builtin such as `open` is bound by no module: it is shadowed on every source module (raising=False).
    Fails when nothing would be patched, so a spy can never pass vacuously."""
    bound = [module for module in _SOURCE_MODULES if name in vars(module)]
    if not bound:
        assert hasattr(builtins, name), f"no audit source module binds {name!r}"
        for module in _SOURCE_MODULES:
            monkeypatch.setattr(module, name, value, raising=False)
        return
    original = vars(bound[0])[name]
    for module in bound:
        if vars(module)[name] is original:
            monkeypatch.setattr(module, name, value)


def _unpatch_bindings(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Undo a builtin shadow set by `_patch_bindings`."""
    for module in _SOURCE_MODULES:
        monkeypatch.delattr(module, name, raising=False)
